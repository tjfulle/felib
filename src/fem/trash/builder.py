from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .block import ElementBlock
from .constants import DISTLOAD
from .constants import GRAVITY
from .constants import PRESSURE
from .constants import TRACTION
from .element import Element
from .model import Model

ArrayLike = NDArray | Sequence


class RegionSelector(ABC):
    @abstractmethod
    def __call__(self, x: Sequence[float], on_boundary: bool) -> bool: ...


class NodeSelector(ABC):
    @abstractmethod
    def __call__(self, node: "NodeInfo") -> bool: ...


class Value(ABC):
    @abstractmethod
    def __call__(self, x: Sequence[float], t: float = 0.0) -> Any: ...


class Constant(Value):
    def __init__(self, value: Any) -> None:
        self.value = value

    def __call__(self, x: Sequence[float], t: float = 0.0) -> Any:
        return self.value


class Vector(ABC):
    @abstractmethod
    def __call__(self, x: Sequence[float], t: float = 0.0) -> list[float]: ...


class ConstantVector(Vector):
    def __init__(self, value: list[float]) -> None:
        self.value = value

    def __call__(self, x: Sequence[float], t: float = 0.0) -> list[float]:
        return self.value


class Matrix(ABC):
    @abstractmethod
    def __call__(self, x: Sequence[float], t: float = 0.0) -> list[list[float]]: ...


class ConstantMatrix(Matrix):
    def __init__(self, value: list[list[float]]) -> None:
        self.value = value

    def __call__(self, x: Sequence[float], t: float = 0.0) -> list[list[float]]:
        return self.value


class BoundaryCondition(ABC):
    @abstractmethod
    def __call__(self, x: Sequence[float], t: float = 0.0) -> list[tuple[int, float]]: ...


class PinnedBoundary(BoundaryCondition):
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self, x: Sequence[float], t: float = 0.0) -> list[tuple[int, float]]:
        ndim = len(x)
        return [(dof, self.value) for dof in range(ndim)]


class RollerBoundary(BoundaryCondition):
    def __init__(self, free_dof: int, value: float = 0.0) -> None:
        self.free_dof = free_dof
        self.value = value

    def __call__(self, x: Sequence[float], t: float = 0.0) -> list[tuple[int, float]]:
        ndim = len(x)
        return [(dof, self.value) for dof in range(ndim) if dof != self.free_dof]


class PointLoad(BoundaryCondition):
    def __init__(self, free_dof: int, value: float = 0.0) -> None:
        self.free_dof = free_dof
        self.value = value

    def __call__(self, x: Sequence[float], t: float = 0.0) -> list[tuple[int, float]]:
        ndim = len(x)
        return [(dof, self.value) for dof in range(ndim) if dof != self.free_dof]


class PeriodicBoundary:
    pass


@dataclass
class NodeInfo:
    node: int
    index: int
    coords: list[float]
    normal: list[float]
    on_boundary: bool = False


@dataclass
class SurfaceInfo:
    block: int
    element: int
    side: int
    normal: NDArray


class Builder:
    def __init__(self, nodes: list[Sequence[int | float]], elements: list[list[int]]) -> None:
        num_node: int = len(nodes)
        max_dim: int = max(len(n[1:]) for n in nodes)
        self.coords: NDArray = np.zeros((num_node, max_dim), dtype=float)
        self.ix_to_node: list[int] = [int(n[0]) for n in nodes]
        self.node_to_ix: dict[int, int] = {}
        for i, node in enumerate(nodes):
            nid, *xc = node
            self.node_to_ix[int(nid)] = i
            self.coords[i, : len(xc)] = [float(x) for x in xc]

        num_elem: int = len(elements)
        max_elem: int = max(len(e[1:]) for e in elements)
        self.connect: NDArray = -np.ones((num_elem, max_elem), dtype=int)
        self.ix_to_elem: list[int] = [int(e[0]) for e in elements]
        self.elem_to_ix: dict[int, int] = {}
        num_elem: int = len(elements)
        for i, element in enumerate(elements):
            self.elem_to_ix[int(element[0])] = i
            for j, node in enumerate(element[1:]):
                if node not in self.node_to_ix:
                    raise ValueError(f"Node {j + 1} of element {i + 1} ({node}) is not defined")
                self.connect[i, j] = self.node_to_ix[node]

        self.surfaces: list[SurfaceInfo] = []
        self.boundary_nodes: list[NodeInfo] = []

        self.element_blocks: dict[str, dict] = {}
        self.bcs: list[dict] = []
        self.rloads: list[dict] = []
        self.sloads: list[dict] = []
        self.dloads: list[dict] = []
        self.cloads: list[dict] = []
        self.mpcs: list[list[int | float]] = []

        # dof_map[node, dof] -> global (model) dof
        self.dof_map: NDArray = np.empty((0, 0), dtype=int)
        self.node_signatures: NDArray = np.empty((0, 0), dtype=int)

        self.blocks: list[ElementBlock] = []

        # E = block_elem_map[b, e] is the model (internal) element number "E" for the
        # eth internal element of block b
        self.block_elem_map: NDArray = np.empty((0, 0), dtype=int)

        # N = block_node_map[b, n] is the model (internal) node number "N" for the
        # nth internal node of block b
        self.block_node_map: NDArray = np.empty((0, 0), dtype=int)

        self.dirichlet_dofs: list[int] = []
        self.dirichlet_vals: list[float] = []

        self.neumann_dofs: list[int] = []
        self.neumann_vals: list[float] = []

        self.robin_loads: list[list] = []
        self.surface_loads: list[list] = []
        self.distributed_loads: list[list] = []

        self.equations: list[list[int | float]] = []

    def element_block(self, *, name: str, element: Element, region: RegionSelector) -> None:
        if name in self.element_blocks:
            raise ValueError(f"Element block {name!r} already defined")
        self.element_blocks[name] = {"element": element, "region": region}

    def boundary(self, *, region: RegionSelector, bc: BoundaryCondition) -> None:
        self.bcs.append({"region": region, "bc": bc})

    def point_load(self, *, region: RegionSelector, value: Vector) -> None:
        self.cloads.append({"region": region, "vector": value})

    def traction(self, *, region: RegionSelector, value: Vector) -> None:
        self.sloads.append({"region": region, "type": TRACTION, "value": value})

    def robin(self, *, region: RegionSelector, u0: Vector, H: Matrix) -> None:
        self.rloads.append({"region": region, "u0": u0, "H": H})

    def gravity(self, *, region: RegionSelector, value: Vector) -> None:
        self.dloads.append({"region": region, "type": GRAVITY, "value": value})

    def distributed_load(self, *, region: RegionSelector, value: Vector) -> None:
        self.dloads.append({"region": region, "type": DISTLOAD, "value": value})

    def pressure(self, *, region: RegionSelector, value: Value) -> None:
        self.sloads.append({"region": region, "type": PRESSURE, "value": value})

    def periodic_boundary(
        self, *, region_a: RegionSelector, region_b: RegionSelector, value: Value
    ) -> None:
        raise NotImplementedError

    def constraint(
        self,
        *,
        nodes: list[int],
        dofs: list[int],
        coeffs: list[float],
        rhs: float = 0.0,
    ) -> None:
        if not (len(nodes) == len(dofs) == len(coeffs)):
            raise ValueError("nodes, dofs, and coeffs must be same length")
        mpc: list[int | float] = []
        for n, d, c in zip(nodes, dofs, coeffs):
            if n not in self.node_to_ix:
                raise ValueError(f"Node {n} is not defined")
            mpc.extend([n, d, c])
        mpc.append(rhs)
        self.mpcs.append(mpc)

    def assemble(self) -> Model:
        self.assemble_blocks()
        self.detect_topology()
        self.build_dof_maps()
        self.build_bcs()
        self.build_loads()
        self.build_equations()

        return Model(
            dof_map=self.dof_map,
            coords=self.coords,
            connect=self.connect,
            blocks=self.blocks,
            num_dof=self.num_dof,
            block_dof_map=self.block_dof_map,
            node_signatures=self.node_signatures,
            index_to_node=self.ix_to_node,
            node_to_index=self.node_to_ix,
            index_to_elem=self.ix_to_elem,
            elem_to_index=self.elem_to_ix,
            block_elem_map=self.block_elem_map,
            block_node_map=self.block_node_map,
            dirichlet_dofs=np.array(self.dirichlet_dofs),
            dirichlet_vals=np.array(self.dirichlet_vals),
            neumann_dofs=np.array(self.neumann_dofs),
            neumann_vals=np.array(self.neumann_vals),
            robin_loads=self.robin_loads,
            surface_loads=self.surface_loads,
            distributed_loads=self.distributed_loads,
            equations=self.equations,
        )

    def build_bcs(self):
        dbcs: dict[int, float] = {}
        nbcs: dict[int, float] = defaultdict(float)
        for ninfo in self.boundary_nodes:
            x = self.coords[ninfo.index]
            for entry in self.bcs:
                region: RegionSelector = entry["region"]
                if region(x, on_boundary=True):
                    bc: BoundaryCondition = entry["bc"]
                    for local_dof, value in bc(x):
                        dof = self.dof_map[ninfo.index, local_dof]
                        dbcs[int(dof)] = float(value)
            for entry in self.cloads:
                region: RegionSelector = entry["region"]
                if region(x, on_boundary=True):
                    vec: Vector = entry["vector"]
                    for i, vi in enumerate(vec(x)):
                        if abs(vi) > 0.0:
                            dof = self.dof_map[ninfo.index, i]
                            if dof >= 0:
                                nbcs[int(dof)] += float(vi)

        self.dirichlet_dofs = list(dbcs.keys())
        self.dirichlet_vals = list(dbcs.values())

        self.neumann_dofs = list(nbcs.keys())
        self.neumann_vals = list(nbcs.values())

    def build_loads(self):
        self.robin_loads.clear()
        self.surface_loads.clear()
        for sinfo in self.surfaces:
            block = self.blocks[sinfo.block]
            p = block.coords[block.connect[sinfo.element]]
            x = block.element.side_centroid(sinfo.side, p)
            for sload in self.sloads:
                region: RegionSelector = sload["region"]
                if region(x.tolist(), on_boundary=True):
                    vec: Vector = sload["value"]
                    row = [sinfo.block, sinfo.element, sinfo.side, sload["type"], *vec(x.tolist())]
                    self.surface_loads.append(row)
            for rload in self.rloads:
                region: RegionSelector = rload["region"]
                if region(x.tolist(), on_boundary=True):
                    u0: Vector = rload["u0"]
                    H: Matrix = rload["H"]
                    self.robin_loads.append(
                        [sinfo.block, sinfo.element, sinfo.side, H(x.tolist()), u0(x.tolist())]
                    )
        self.surface_loads.sort(key=lambda x: (x[0], x[1], x[2]))
        self.robin_loads.sort(key=lambda x: (x[0], x[1], x[2]))

        self.distributed_loads.clear()
        for b, block in enumerate(self.blocks):
            for e, conn in enumerate(block.connect):
                p = block.coords[conn]
                x = p.mean(axis=0)
                for dload in self.dloads:
                    region: RegionSelector = dload["region"]
                    if region(x, on_boundary=False):
                        vec: Vector = dload["value"]
                        self.distributed_loads.append([b, e, dload["type"], *vec(x)])
        self.distributed_loads.sort(key=lambda x: (x[0], x[1]))

    def build_equations(self) -> None:
        self.equations.clear()
        for mpc in self.mpcs:
            rhs = mpc[-1]
            equation: list[int | float] = []
            for i in range(0, len(mpc[:-1]), 3):
                node = int(mpc[i])
                dof = int(mpc[i + 1])
                coeff = mpc[i + 2]
                equation.extend((self.node_to_ix[node], dof, coeff))
            equation.append(rhs)
            self.equations.append(equation)

    def assemble_blocks(self) -> None:
        self.blocks.clear()
        blk_ele_map: list[NDArray] = []
        blk_nod_map: list[NDArray] = []

        max_node_in_any_block = -1
        max_elem_in_any_block = -1

        for name, decl in self.element_blocks.items():
            ids: list[int] = []
            region: RegionSelector = decl["region"]
            element: Element = decl["element"]
            for e, conn in enumerate(self.connect):
                p = self.coords[conn]
                x = p.mean(axis=0)
                if region(x, on_boundary=False):
                    ids.append(e)

            # By this point, ids holds the global IDs of each element in the block
            elements = np.asarray(ids, dtype=int)
            assigned_elements = [e for blk in blk_ele_map for e in blk]
            mask = np.isin(elements, assigned_elements)
            if np.any(mask):
                duplicates = ", ".join(str(e) for e in elements[mask])
                raise ValueError(
                    f"Block {name}: attempting to assign elements {duplicates} "
                    "which are already assigned to other element blocks"
                )

            nodes_per_elem = len(element.node_freedom_table)
            connect = self.connect[elements, :nodes_per_elem]
            if np.any(connect == -1):
                raise ValueError(
                    f"Block {name}: wrong number of nodes in one or more elements "
                    f"(elements in this block must have {nodes_per_elem} nodes"
                )

            blk_ele_map.append(elements)
            max_elem_in_any_block = max(max_elem_in_any_block, len(elements))

            # Connect contains the node IDs for the block connectivity in the model system
            # Convert to local block IDs
            nodes, inverse = np.unique(connect, return_inverse=True)
            xb = self.coords[nodes]
            cb = inverse.reshape(connect.shape)
            block = ElementBlock(name, xb, cb, element)
            self.blocks.append(block)
            blk_nod_map.append(nodes)
            max_node_in_any_block = max(max_node_in_any_block, len(nodes))

        num_blocks = len(self.blocks)
        self.block_elem_map = -np.ones((num_blocks, max_elem_in_any_block), dtype=int)
        for b, elements in enumerate(blk_ele_map):
            self.block_elem_map[b, : len(elements)] = elements

        # Check if all elements are assigned to an element block
        assigned_elements = self.block_elem_map[np.where(self.block_elem_map != -1)]
        num_elem = len(self.elem_to_ix)
        if unassigned := set(range(num_elem)).difference(assigned_elements):
            s = ", ".join(str(_) for _ in unassigned)
            raise ValueError(f"Elements {s} not assigned to any element blocks")

        # Check that all nodes belong to an element or have a Dirichlet BC
        connected: set[int] = set([_ for _ in self.connect.flatten() if _ != -1])
        allnodes: set[int] = set(range(self.coords.shape[0]))
        if disconnected := allnodes.difference(connected):
            for n in disconnected:
                node = self.ix_to_node[n]
                raise ValueError(
                    f"Node {node} is not connected to any element "
                    "and does not have an associated dirichlet BC."
                )

        self.block_node_map = -np.ones((num_blocks, max_node_in_any_block), dtype=int)
        for b, nodes in enumerate(blk_nod_map):
            self.block_node_map[b, : len(nodes)] = nodes

    def detect_topology(self) -> None:
        """
        Detect boundary faces/edges for all blocks and elements.

        Returns:
            surfaces: list of tuples (block_id, local_elem_id, local_face_id)
        """
        # mapping from face (tuple of sorted node indices) -> list of (block_id, local_elem_id, local_face_id)
        sides: dict[tuple[int, ...], list[tuple[int, int, int]]] = defaultdict(list)

        # Step 1: iterate all blocks and all elements in each block
        for b, block in enumerate(self.blocks):
            for e, conn in enumerate(block.connect):
                for s, side in enumerate(block.element.sides):
                    nodes = conn[side]
                    global_nodes = tuple(sorted([self.block_node_map[b, n] for n in nodes]))
                    sides[global_nodes].append((b, e, s))

        # Step 2: identify faces that are only in one element → boundary
        self.surfaces.clear()
        edge_normals: dict[int, list[NDArray]] = defaultdict(list)
        for global_nodes, elems in sides.items():
            if len(elems) == 1:
                b, e, s = elems[0]
                block = self.blocks[b]
                conn = block.connect[e]
                p = block.coords[conn]
                normal = block.element.edge_normal(s, p, xi=0.0)
                normal /= np.linalg.norm(normal)
                si = SurfaceInfo(block=b, element=e, side=s, normal=normal)
                self.surfaces.append(si)
                for ln in block.element.sides[s]:
                    gn = self.block_node_map[b, conn[ln]]
                    edge_normals[gn].append(normal)

        self.boundary_nodes.clear()
        for gn, normals in edge_normals.items():
            avg_normal = np.mean(normals, axis=0)
            id = self.ix_to_node[gn]
            ni = NodeInfo(
                coords=self.coords[gn].tolist(),
                node=int(id),
                index=int(gn),
                normal=avg_normal.tolist(),
                on_boundary=True,
            )
            self.boundary_nodes.append(ni)
        return

    def build_dof_maps(self) -> None:
        # Build node signatures and dof maps
        num_node: int = self.coords.shape[0]
        max_dof_per_node = max([b.element.dof_per_node for b in self.blocks])
        # FIX ME: compress out dofs not used at all
        max_dof_per_node = max(max_dof_per_node, 10)
        self.node_signatures = np.zeros((num_node, max_dof_per_node), dtype=int)
        for b, block in enumerate(self.blocks):
            node_freedoms = np.asarray(block.element.node_freedom_table, dtype=int)
            for nodes in block.connect:
                # Map block local node number to model node number
                ix = [self.block_node_map[b, node] for node in nodes]
                self.node_signatures[ix, :] |= node_freedoms

        # Check for dummy nodes that are not included in element connectivity
        # All dummy nodes must have associated Dirichlet BCs.  We can use the BC to fill in the node
        # signature
        # for bc in self.bcs:
        #    self.node_signatures[bc["nodes"], bc["local_dof"]] |= 1
        if np.any(self.node_signatures.sum(axis=1) == 0):
            missing = np.where(self.node_signatures.sum(axis=1) == 0)[0]
            raise ValueError(f"Dummy node without Dirichlet BC: {missing.tolist()}")

        active_mask: NDArray = self.node_signatures == 1
        self.num_dof: int = int(active_mask.sum())

        mask = active_mask.ravel()
        map = -np.ones_like(mask, dtype=int)
        map[mask] = np.arange(self.num_dof)
        self.dof_map = map.reshape(self.node_signatures.shape).copy()

        # Create map
        # DOF = block_dof_map[b, dof] is the global model DOF for dof of block b
        m = max([block.num_dof for block in self.blocks])
        self.block_dof_map = -np.ones((len(self.blocks), m), dtype=int)
        for b, block in enumerate(self.blocks):
            nodes = np.unique(block.connect)
            for node in nodes:
                N = self.block_node_map[b, node]
                for local_dof in range(block.element.dof_per_node):
                    dof = block.dof_map[node, local_dof]
                    DOF = self.dof_map[N, local_dof]
                    self.block_dof_map[b, dof] = DOF
