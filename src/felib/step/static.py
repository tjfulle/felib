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

    def compile(
        self, model: "Model", dof_manager: "DOFManager", parent: CompiledStep | None
    ) -> CompiledStep:
        return CompiledStaticStep(
            name=self.name,
            parent=parent,
            period=self.period,
            dbcs=self.compile_dbcs(model, dof_manager),
            nbcs=self.compile_nbcs(model, dof_manager),
            dloads=self.compile_dloads(model),
            dsloads=self.compile_dsloads(model),
            rloads=self.compile_rloads(model),
            equations=self.compile_constraints(model, dof_manager),
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
        for nodes, dofs, coeffs, rhs in self.metadata.get("constraints", {}).values():
            mpc: list[int | float] = []
            for gid, dof, coeff in zip(nodes, dofs, coeffs):
                if gid not in model.node_map:
                    raise ValueError(f"Node {gid} is not defined")
                lid = model.node_map.local(gid)
                DOF = dof_manager.global_dof(lid, dof)
                mpc.extend([DOF, coeff])
            mpc.append(rhs)
            mpcs.append(mpc)
        return mpcs


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
        fdofs = np.array(sorted(set(range(ndof)) - set(ddofs)))
        nf = len(fdofs)
        neq = len(self.equations) if self.equations else 0

        x0 = u0[fdofs]
        if neq > 0:
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
        )
        solver = NonlinearNewtonSolver()
        state = solver(
            kernel,
            x0,
            atol=self.solver_options.get("atol"),
            rtol=self.solver_options.get("rtol"),
            maxiter=self.solver_options.get("maxiter"),
        )
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
