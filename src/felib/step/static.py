from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..collections import DistributedLoad
from ..collections import DistributedSurfaceLoad
from ..collections import Field
from ..collections import GravityLoad
from ..collections import NodeData
from ..collections import PressureLoad
from ..collections import RobinLoad
from ..collections import Solution
from ..collections import TractionLoad
from ..constants import NodeVariable
from ..constants import SpatialVectorVar
from ..solver import NonlinearNewtonSolver
from ..typing import DLoadT
from ..typing import DSLoadT
from ..typing import RLoadT
from .assemble import AssemblyKernel
from .base import CompiledStep
from .base import Step

if TYPE_CHECKING:
    from ..dof_manager import DOFManager
    from ..model import Model


class StaticStep(Step):
    def __init__(self, name: str, ndim: int, period: float = 1.0, **options: Any) -> None:
        super().__init__(name=name, ndim=ndim, period=period)
        self.auto_tie_interfaces = bool(options.pop("auto_tie_interfaces", True))
        self.solver_opts = options

    def boundary(
        self, *, nodes: str | int | list[int], dofs: int | list[int], value: float = 0.0
    ) -> None:
        if isinstance(dofs, int):
            dofs = [dofs]
        dbcs = self.metadata["dbcs"]
        dbcs[f"dbc-{len(dbcs)}"] = (nodes, dofs, value)

    def point_load(
        self, *, nodes: str | int | list[int], dofs: int | list[int], value: float = 0.0
    ) -> None:
        if isinstance(dofs, int):
            dofs = [dofs]
        nbcs = self.metadata["nbcs"]
        nbcs[f"nbc-{len(nbcs)}"] = (nodes, dofs, value)

    def traction(self, *, sideset: str, magnitude: float, direction: Sequence[float]) -> None:
        dsloads = self.metadata["dsloads"]
        dir = normalize(direction)
        dsloads[f"dsload-{len(dsloads)}"] = ("traction", sideset, magnitude, dir)

    def pressure(self, *, sideset: str, magnitude: float) -> None:
        dsloads = self.metadata["dsloads"]
        dsloads[f"dsload-{len(dsloads)}"] = ("pressure", sideset, magnitude)

    def gravity(
        self, *, elements: str | int | list[int], g: float, direction: Sequence[float]
    ) -> None:
        dloads = self.metadata["dloads"]
        dir = normalize(direction)
        dloads[f"dload-{len(dloads)}"] = ("gravity", elements, g, dir)

    def dload(self, *, elements: str | int | list[int], field: Field) -> None:
        dloads = self.metadata["dloads"]
        dloads[f"dload-{len(dloads)}"] = ("dload", elements, field)

    def robin(self, *, sideset: str, u0: NDArray, H: NDArray) -> None:
        rloads = self.metadata["rloads"]
        rloads[f"rload-{len(rloads)}"] = (sideset, H, u0)

    def equation(self, *args: int | float) -> None:
        if len(args) < 4:
            raise ValueError("Equation at least one (node, dof, coeff) triple and rhs")
        if (len(args) - 1) % 3 != 0:
            raise ValueError("Equation must be (node, dof, coeff), ..., rhs")
        constraints = self.metadata["constraints"]
        triples = args[:-1]
        rhs = args[-1]
        nodes: list[int] = []
        dofs: list[int] = []
        coeffs: list[float] = []
        for i in range(0, len(triples), 3):
            nodes.append(int(triples[i]))
            dofs.append(int(triples[i + 1]))
            coeffs.append(float(triples[i + 2]))
        constraints[f"constraint-{len(constraints)}"] = (nodes, dofs, coeffs, rhs)

    def tie_nodes(
        self,
        pairs: Sequence[Sequence[int]] | None = None,
        *,
        secondary_nodes: int | Sequence[int] | None = None,
        primary_nodes: int | Sequence[int] | None = None,
        dofs: int | Sequence[int] | None = None,
    ) -> None:
        """
        Tie manually specified node pairs with homogeneous equation constraints.

        ``pairs`` and ``secondary_nodes``/``primary_nodes`` use
        ``(secondary_node, primary_node)`` ordering.
        """
        if pairs is not None and (secondary_nodes is not None or primary_nodes is not None):
            raise ValueError("Expected pairs or secondary_nodes/primary_nodes, not both")
        if pairs is None:
            if secondary_nodes is None or primary_nodes is None:
                raise ValueError("Expected pairs or both secondary_nodes and primary_nodes")
            secondary = _as_list(secondary_nodes)
            primary = _as_list(primary_nodes)
            if len(secondary) != len(primary):
                raise ValueError("secondary_nodes and primary_nodes must have the same length")
            pairs = list(zip(secondary, primary))

        tie_dofs = self._tie_dofs(dofs)
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError("Each tied node pair must be (secondary_node, primary_node)")
            secondary, primary = int(pair[0]), int(pair[1])
            for dof in tie_dofs:
                self.equation(secondary, dof, 1.0, primary, dof, -1.0, 0.0)

    def tied_nodes(self, *args: Any, **kwargs: Any) -> None:
        self.tie_nodes(*args, **kwargs)

    def tie_matching_nodes(
        self,
        *,
        secondary_nodes: str | int | Sequence[int],
        primary_nodes: str | int | Sequence[int],
        dofs: int | Sequence[int] | None = None,
        tol: float = 1e-12,
    ) -> None:
        """
        Tie nodes by matching coordinates between secondary and primary sets.

        Nodes without a coordinate match are ignored, which lets callers pass
        entire mirrored mesh halves while only coincident interface nodes are
        constrained.
        """
        if tol < 0.0:
            raise ValueError("tol must be non-negative")
        ties = self.metadata["tied_node_matches"]
        ties[f"tied-node-match-{len(ties)}"] = (
            secondary_nodes,
            primary_nodes,
            self._tie_dofs(dofs),
            float(tol),
        )

    def tie_nodes_to_surface(
        self,
        *,
        secondary_nodes: str | int | Sequence[int],
        primary_sides: str | Sequence[Sequence[int]],
        dofs: int | Sequence[int] | None = None,
        tol: float = 1e-8,
    ) -> None:
        """
        Tie secondary nodes to a nonconforming primary surface.

        Each secondary node is projected onto the nearest primary side and
        constrained to the primary side's nodal interpolation. ``primary_sides``
        may be a sideset name or explicit ``(element_gid, side_id)`` pairs where
        ``side_id`` is 1-based, matching :meth:`Mesh.sideset`.
        """
        if tol < 0.0:
            raise ValueError("tol must be non-negative")
        ties = self.metadata["tied_node_surfaces"]
        ties[f"tied-node-surface-{len(ties)}"] = (
            secondary_nodes,
            primary_sides,
            self._tie_dofs(dofs),
            float(tol),
        )

    def tie_nonmatching_nodes(self, *args: Any, **kwargs: Any) -> None:
        self.tie_nodes_to_surface(*args, **kwargs)

    def _tie_dofs(self, dofs: int | Sequence[int] | None) -> list[int]:
        if dofs is None:
            return list(range(self.ndim))
        if isinstance(dofs, int):
            return [int(dofs)]
        return [int(dof) for dof in dofs]

    def compile(
        self, model: "Model", dof_manager: "DOFManager", parent: CompiledStep | None
    ) -> CompiledStep:
        equations = self.compile_constraints(model, dof_manager)
        self._configure_mpc_map(dof_manager, equations)
        return CompiledStaticStep(
            name=self.name,
            parent=parent,
            period=self.period,
            dbcs=self.compile_dbcs(model, dof_manager),
            nbcs=self.compile_nbcs(model, dof_manager),
            dloads=self.compile_dloads(model),
            dsloads=self.compile_dsloads(model),
            rloads=self.compile_rloads(model),
            equations=equations,
            solver_options=self.solver_opts,
        )

    def node_variables(self) -> list[NodeVariable]:
        variables: list[NodeVariable] = [SpatialVectorVar("u"), SpatialVectorVar("f")]
        return variables

    def compile_dbcs(self, model: "Model", dof_manager: "DOFManager") -> list[tuple[int, float]]:
        seen: dict[int, float] = {}
        for nodes, dofs, value in self.metadata.get("dbcs", {}).values():
            lids: list[int]
            if isinstance(nodes, str):
                if nodes not in model.nodesets:
                    raise ValueError(f"nodeset {nodes} not defined")
                lids = model.nodesets[nodes]
            elif isinstance(nodes, int):
                lids = [model.node_map.local(nodes)]
            else:
                lids = [model.node_map.local(gid) for gid in nodes]
            for lid in lids:
                for dof in dofs:
                    DOF = dof_manager.global_dof(lid, dof)
                    seen[DOF] = value
        dbcs = [(k, seen[k]) for k in sorted(seen)]
        return dbcs

    def compile_nbcs(self, model: "Model", dof_manager: "DOFManager") -> list[tuple[int, float]]:
        seen: dict[int, float] = defaultdict(float)
        for nodes, dofs, value in self.metadata.get("nbcs", {}).values():
            lids: list[int]
            if isinstance(nodes, str):
                if nodes not in model.nodesets:
                    raise ValueError(f"nodeset {nodes} not defined")
                lids = model.nodesets[nodes]
            elif isinstance(nodes, int):
                lids = [model.node_map.local(nodes)]
            else:
                lids = [model.node_map.local(gid) for gid in nodes]
            for lid in lids:
                for dof in dofs:
                    DOF = dof_manager.global_dof(lid, dof)
                    seen[DOF] += value
        nbcs = [(k, seen[k]) for k in sorted(seen)]
        return nbcs

    def compile_dloads(self, model: "Model") -> DLoadT:
        dloads: DLoadT = defaultdict(lambda: defaultdict(list))
        for ltype, elements, *args in self.metadata.get("dloads", {}).values():
            dload: DistributedLoad | None = None
            lids: list[int]
            if isinstance(elements, str):
                if elements not in model.elemsets:
                    raise ValueError(f"element set {elements} not defined")
                lids = model.elemsets[elements]
            elif isinstance(elements, int):
                lids = [model.elem_map.local(elements)]
            else:
                lids = [model.elem_map.local(gid) for gid in elements]
            if ltype == "gravity":
                pass
            elif ltype == "dload":
                field = args[0]
                dload = DistributedLoad(field=field)
            else:
                raise ValueError(f"Unknown ltype: {ltype}")
            for lid in lids:
                gid = model.elem_map[lid]
                block_no = model.block_elem_map[lid]
                block = model.blocks[block_no]
                if ltype == "gravity":
                    g, direction = args
                    dload = GravityLoad(block.material.density * g, direction)
                e = block.elem_map.local(gid)
                assert dload is not None
                dloads[block_no][e].append(dload)
        return dloads

    def compile_dsloads(self, model: "Model") -> DSLoadT:
        dsloads: DSLoadT = defaultdict(lambda: defaultdict(list))
        for ltype, sideset, *args in self.metadata.get("dsloads", {}).values():
            dsload: DistributedSurfaceLoad
            if sideset not in model.sidesets:
                raise ValueError(f"side set {sideset} not defined")
            if ltype == "traction":
                magnitude, direction = args
                dsload = TractionLoad(magnitude=magnitude, direction=direction)
            elif ltype == "pressure":
                magnitude = args[0]
                dsload = PressureLoad(magnitude=magnitude)
            else:
                raise ValueError(f"Unknown ltype: {ltype}")
            for elem_no, edge_no in model.sidesets[sideset]:
                block_no = model.block_elem_map[elem_no]
                block = model.blocks[block_no]
                gid = model.elem_map[elem_no]
                lid = block.elem_map.local(gid)
                dsloads[block_no][lid].append((edge_no, dsload))
        return dsloads

    def compile_rloads(self, model: "Model") -> RLoadT:
        sideset: str
        H: NDArray
        u0: NDArray
        rloads: RLoadT = defaultdict(lambda: defaultdict(list))
        for sideset, H, u0 in self.metadata.get("rloads", {}).values():
            if sideset not in model.sidesets:
                raise ValueError(f"side set {sideset} not defined")
            for ele_no, edge_no in model.sidesets[sideset]:
                block_no = model.block_elem_map[ele_no]
                block = model.mesh.blocks[block_no]
                gid = block.elem_map[ele_no]
                lid = block.elem_map.local(gid)
                rload = RobinLoad(edge=edge_no, H=H, u0=u0)
                rloads[block_no][lid].append(rload)
        return rloads

    def compile_constraints(
        self, model: "Model", dof_manager: "DOFManager"
    ) -> list[list[int | float]]:
        nodes: list[int]
        dofs: list[int]
        coeffs: list[float]
        rhs: float = 0.0
        mpcs: list[list[int | float]] = []
        seen: set[tuple] = set()
        for nodes, dofs, coeffs, rhs in self.metadata.get("constraints", {}).values():
            self._append_unique_constraint(
                mpcs,
                seen,
                self._compile_constraint(model, dof_manager, nodes, dofs, coeffs, rhs),
            )
        for secondary_nodes, primary_nodes, dofs, tol in self.metadata.get(
            "tied_node_matches", {}
        ).values():
            pairs = self._matching_tied_node_pairs(
                model,
                secondary_nodes=secondary_nodes,
                primary_nodes=primary_nodes,
                tol=tol,
            )
            for secondary, primary in pairs:
                for dof in dofs:
                    self._append_unique_constraint(
                        mpcs,
                        seen,
                        self._compile_constraint(
                            model,
                            dof_manager,
                            [secondary, primary],
                            [dof, dof],
                            [1.0, -1.0],
                            0.0,
                        ),
                    )
        for secondary_nodes, primary_sides, dofs, tol in self.metadata.get(
            "tied_node_surfaces", {}
        ).values():
            for secondary, primary_gids, weights in self._node_surface_ties(
                model,
                secondary_nodes=secondary_nodes,
                primary_sides=primary_sides,
                tol=tol,
            ):
                for dof in dofs:
                    nodes, coeffs = self._node_surface_constraint_terms(
                        secondary,
                        primary_gids,
                        weights,
                    )
                    if not nodes:
                        continue
                    self._append_unique_constraint(
                        mpcs,
                        seen,
                        self._compile_constraint(
                            model,
                            dof_manager,
                            nodes,
                            [dof] * len(nodes),
                            coeffs,
                            0.0,
                        ),
                    )
        if self.auto_tie_interfaces:
            for secondary, primary in model.mesh.tied_node_pairs:
                for dof in self._tie_dofs(None):
                    self._append_unique_constraint(
                        mpcs,
                        seen,
                        self._compile_constraint(
                            model,
                            dof_manager,
                            [secondary, primary],
                            [dof, dof],
                            [1.0, -1.0],
                            0.0,
                        ),
                    )
        for constraint in model.constraints:
            self._append_unique_constraint(
                mpcs,
                seen,
                self._compile_mpc_constraint(constraint),
            )
        return mpcs

    def _configure_mpc_map(
        self,
        dof_manager: "DOFManager",
        equations: list[list[int | float]],
    ) -> None:
        dof_manager.clear_mpc_map()
        if not equations:
            return
        if not self._homogeneous_constraints(equations):
            return
        try:
            dof_manager.build_mpc_map(equations)
        except (RuntimeError, ValueError):
            dof_manager.clear_mpc_map()

    @staticmethod
    def _homogeneous_constraints(
        equations: list[list[int | float]],
        *,
        tol: float = 1e-14,
    ) -> bool:
        return all(abs(float(equation[-1])) <= tol for equation in equations)

    def _append_unique_constraint(
        self,
        mpcs: list[list[int | float]],
        seen: set[tuple],
        mpc: list[int | float],
    ) -> None:
        key = self._constraint_key(mpc)
        if key in seen:
            return
        seen.add(key)
        mpcs.append(mpc)

    def _constraint_key(self, mpc: list[int | float]) -> tuple:
        rhs = float(mpc[-1])
        terms = [(int(mpc[i]), float(mpc[i + 1])) for i in range(0, len(mpc) - 1, 2)]
        terms.sort(key=lambda item: item[0])
        if terms:
            first_coeff = terms[0][1]
            if first_coeff < 0.0:
                terms = [(dof, -coeff) for dof, coeff in terms]
                rhs = -rhs
        return tuple(terms + [("rhs", rhs)])

    def _compile_mpc_constraint(self, constraint) -> list[int | float]:
        slave_dof, masters, offset = constraint.to_tuple()
        mpc: list[int | float] = [slave_dof, 1.0]
        for master_dof, coeff in masters:
            mpc.extend([master_dof, -coeff])
        mpc.append(offset)
        return mpc

    def _compile_constraint(
        self,
        model: "Model",
        dof_manager: "DOFManager",
        nodes: Sequence[int],
        dofs: Sequence[int],
        coeffs: Sequence[float],
        rhs: float,
    ) -> list[int | float]:
        mpc: list[int | float] = []
        for gid, dof, coeff in zip(nodes, dofs, coeffs):
            if gid not in model.node_map:
                raise ValueError(f"Node {gid} is not defined")
            lid = model.node_map.local(gid)
            DOF = dof_manager.global_dof(lid, dof)
            mpc.extend([DOF, coeff])
        mpc.append(rhs)
        return mpc

    def _matching_tied_node_pairs(
        self,
        model: "Model",
        *,
        secondary_nodes: str | int | Sequence[int],
        primary_nodes: str | int | Sequence[int],
        tol: float,
    ) -> list[tuple[int, int]]:
        primary_gids = self._node_gids(model, primary_nodes)
        secondary_gids = self._node_gids(model, secondary_nodes)
        primary = [(gid, model.coords[model.node_map.local(gid)]) for gid in primary_gids]
        pairs: list[tuple[int, int]] = []
        for secondary_gid in secondary_gids:
            secondary_coord = model.coords[model.node_map.local(secondary_gid)]
            matches = [
                primary_gid
                for primary_gid, primary_coord in primary
                if np.allclose(secondary_coord, primary_coord, atol=tol, rtol=0.0)
            ]
            if len(matches) > 1:
                raise ValueError(
                    f"Secondary node {secondary_gid} matches multiple primary nodes: {matches}"
                )
            if len(matches) == 1 and matches[0] != secondary_gid:
                pairs.append((secondary_gid, matches[0]))
        if not pairs:
            raise ValueError("No tied node pairs with matching coordinates were found")
        return pairs

    def _node_gids(self, model: "Model", nodes: str | int | Sequence[int]) -> list[int]:
        if isinstance(nodes, str):
            if nodes not in model.nodesets:
                raise ValueError(f"nodeset {nodes} not defined")
            gids = [model.node_map[lid] for lid in model.nodesets[nodes]]
        elif isinstance(nodes, int):
            gids = [int(nodes)]
        else:
            gids = [int(gid) for gid in nodes]
        for gid in gids:
            if gid not in model.node_map:
                raise ValueError(f"Node {gid} is not defined")
        return list(dict.fromkeys(gids))

    def _node_surface_ties(
        self,
        model: "Model",
        *,
        secondary_nodes: str | int | Sequence[int],
        primary_sides: str | Sequence[Sequence[int]],
        tol: float,
    ) -> list[tuple[int, list[int], NDArray]]:
        secondary_gids = self._node_gids(model, secondary_nodes)
        side_specs = self._side_specs(model, primary_sides)
        ties: list[tuple[int, list[int], NDArray]] = []
        for secondary_gid in secondary_gids:
            x = model.coords[model.node_map.local(secondary_gid)]
            best: tuple[float, list[int], NDArray] | None = None
            for elem_lid, side_no in side_specs:
                candidate = self._project_to_primary_side(
                    model,
                    elem_lid=elem_lid,
                    side_no=side_no,
                    x=x,
                    tol=tol,
                )
                if candidate is None:
                    continue
                distance, primary_gids, weights = candidate
                if best is None or distance < best[0]:
                    best = (distance, primary_gids, weights)
            if best is None:
                raise ValueError(
                    f"Secondary node {secondary_gid} could not be projected onto "
                    f"the primary surface within tol={tol}"
                )
            _, primary_gids, weights = best
            ties.append((secondary_gid, primary_gids, weights))
        return ties

    def _side_specs(
        self,
        model: "Model",
        primary_sides: str | Sequence[Sequence[int]],
    ) -> list[tuple[int, int]]:
        if isinstance(primary_sides, str):
            if primary_sides not in model.sidesets:
                raise ValueError(f"side set {primary_sides} not defined")
            return list(model.sidesets[primary_sides])

        side_specs: list[tuple[int, int]] = []
        for side in primary_sides:
            if len(side) != 2:
                raise ValueError("Primary sides must be (element_gid, side_id) pairs")
            elem_gid = int(side[0])
            side_id = int(side[1])
            if elem_gid not in model.elem_map:
                raise ValueError(f"Element {elem_gid} is not defined")
            elem_lid = model.elem_map.local(elem_gid)
            block_no = model.block_elem_map[elem_lid]
            ref_el = model.mesh.blocks[block_no].ref_el
            if side_id < 1 or side_id > ref_el.nedge:
                raise ValueError(
                    f"Side {side_id} is invalid for element {elem_gid}; expected 1..{ref_el.nedge}"
                )
            side_specs.append((elem_lid, side_id - 1))
        if not side_specs:
            raise ValueError("primary_sides is empty")
        return side_specs

    def _project_to_primary_side(
        self,
        model: "Model",
        *,
        elem_lid: int,
        side_no: int,
        x: NDArray,
        tol: float,
    ) -> tuple[float, list[int], NDArray] | None:
        block_no = model.block_elem_map[elem_lid]
        block = model.mesh.blocks[block_no]
        elem_gid = model.elem_map[elem_lid]
        block_elem_lid = block.elem_map.local(elem_gid)
        conn = block.connect[block_elem_lid]
        edge_ix = block.ref_el.edge_nodes(side_no)
        primary_gids = [block.node_map[int(node_lid)] for node_lid in conn[edge_ix]]
        primary_coords = np.array(
            [model.coords[model.node_map.local(gid)] for gid in primary_gids],
            dtype=float,
        )
        projection = self._project_point_to_edge(
            primary_coords,
            x,
            block.ref_el,
            tol=tol,
        )
        if projection is None:
            return None
        xi, distance = projection
        weights = block.ref_el.edge_shape(xi, len(primary_gids))
        return distance, primary_gids, weights

    def _project_point_to_edge(
        self,
        edge_coords: NDArray,
        x: NDArray,
        ref_el,
        *,
        tol: float,
    ) -> tuple[float, float] | None:
        if len(edge_coords) == 2:
            p0, p1 = edge_coords
            tangent = p1 - p0
            length2 = float(np.dot(tangent, tangent))
            if length2 <= 0.0:
                raise ValueError("Primary side has zero length")
            t = float(np.dot(x - p0, tangent) / length2)
            t_clamped = min(1.0, max(0.0, t))
            xi = 2.0 * t_clamped - 1.0
            closest = p0 + t_clamped * tangent
            distance = float(np.linalg.norm(x - closest))
            if distance <= tol:
                return xi, distance
            return None

        sample_xis = np.linspace(-1.0, 1.0, 9)
        distances = [
            np.linalg.norm(np.dot(ref_el.edge_shape(float(xi), len(edge_coords)), edge_coords) - x)
            for xi in sample_xis
        ]
        xi = float(sample_xis[int(np.argmin(distances))])
        for _ in range(25):
            shape = ref_el.edge_shape(xi, len(edge_coords))
            dshape = ref_el.edge_shape_derivative(xi, len(edge_coords))
            closest = np.dot(shape, edge_coords)
            tangent = np.dot(dshape, edge_coords)
            denom = float(np.dot(tangent, tangent))
            if denom <= 0.0:
                raise ValueError("Primary side has zero tangent during projection")
            delta = float(np.dot(closest - x, tangent) / denom)
            next_xi = float(np.clip(xi - delta, -1.0, 1.0))
            if abs(next_xi - xi) <= max(tol, 1e-12):
                xi = next_xi
                break
            xi = next_xi

        closest = np.dot(ref_el.edge_shape(xi, len(edge_coords)), edge_coords)
        distance = float(np.linalg.norm(x - closest))
        if distance <= tol:
            return xi, distance
        return None

    @staticmethod
    def _node_surface_constraint_terms(
        secondary: int,
        primary_gids: Sequence[int],
        weights: Sequence[float],
        *,
        tol: float = 1e-14,
    ) -> tuple[list[int], list[float]]:
        terms: dict[int, float] = {int(secondary): 1.0}
        for primary_gid, weight in zip(primary_gids, weights):
            gid = int(primary_gid)
            terms[gid] = terms.get(gid, 0.0) - float(weight)
        nodes: list[int] = []
        coeffs: list[float] = []
        for gid, coeff in terms.items():
            if abs(coeff) <= tol:
                continue
            nodes.append(gid)
            coeffs.append(coeff)
        return nodes, coeffs


@dataclass
class CompiledStaticStep(CompiledStep):
    solver_options: dict[str, Any] = field(default_factory=dict)

    def solve(
        self,
        fun: Callable[..., tuple[NDArray, NDArray]],
        u0: NDArray,
        ndata: NodeData,
        args: tuple[Any, ...] = (),
    ) -> NDArray:
        ddofs = self.ddofs
        ndof = len(u0)
        dof_manager = args[0] if len(args) > 0 and hasattr(args[0], "step_transform") else None
        use_mpc_reduction = (
            dof_manager is not None
            and getattr(dof_manager, "has_mpc_transform", False)
            and dof_manager.can_apply_mpc_reduction(ddofs)
        )
        neq = len(self.equations) if self.equations else 0

        if use_mpc_reduction:
            x0 = dof_manager.reduced_initial_values(u0, ddofs)
        else:
            fdofs = np.array(sorted(set(range(ndof)) - set(ddofs)))
            x0 = u0[fdofs]
        nf = len(x0)
        if neq > 0 and not use_mpc_reduction:
            x0 = np.hstack([x0, np.zeros(neq)])

        increment = 1
        time = (0, self.start)
        dt = self.period
        kernel = AssemblyKernel(
            fun,
            u0,
            args=args,
            step=self.number,
            increment=increment,
            time=time,
            dt=dt,
            ddofs=ddofs,
            dvals=self.dvals[1, :],
            nbcs=self.nbcs,
            dloads=self.dloads,
            dsloads=self.dsloads,
            rloads=self.rloads,
            equations=self.equations,
            dof_manager=dof_manager,
        )
        solver = NonlinearNewtonSolver()
        state = solver(
            kernel,
            x0,
            atol=self.solver_options.get("atol"),
            rtol=self.solver_options.get("rtol"),
            maxiter=self.solver_options.get("maxiter"),
        )
        if use_mpc_reduction:
            u = dof_manager.expand_step_solution(state.x[:nf], ddofs, self.dvals[1, :])
        else:
            u = u0.copy()
            u[fdofs] = state.x[:nf]
            u[ddofs] = self.dvals[1, :]

        R = kernel.resid
        K = kernel.stiff
        react = np.zeros_like(R)
        react[ddofs] = R[ddofs]
        self.solution = Solution(
            stiff=K[:ndof, :ndof],
            force=R[:ndof],
            dofs=u[:ndof],
            react=react,
            lagrange_multipliers=state.x[nf:],
            iterations=state.iterations,
        )

        f = np.dot(K[:ndof, :ndof], u[:ndof])
        return u[:ndof]


#        ndata[NodeVar.Fx] = f[0::2]
#        ndata[NodeVar.Fy] = f[1::2]


def normalize(a: Sequence[float]) -> NDArray:
    v = np.asarray(a)
    return v / np.linalg.norm(v)


def _as_list(value: int | Sequence[int]) -> list[int]:
    if isinstance(value, int):
        return [int(value)]
    return [int(item) for item in value]
