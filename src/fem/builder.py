import logging
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from typing import Any
from typing import Sequence
from typing import Type

import numpy as np
from numpy.typing import NDArray

from . import collections
from .block import ElementBlock
from .block import TopoBlock
from .cell import Cell
from .collections import DistributedLoad
from .collections import DistributedSurfaceLoad
from .collections import Field
from .collections import GravityLoad
from .collections import HeatFlux
from .collections import HeatSource
from .collections import PressureLoad
from .collections import RobinLoad
from .collections import TractionLoad
from .element import Element
from .material import Material
from .mesh import Mesh
from .model import Model
from .step import DirectStep
from .step import HeatTransferStep
from .step import StaticStep
from .step import Step
from .typing import DLoadT
from .typing import DSLoadT
from .typing import HLoadT
from .typing import QLoadT
from .typing import RegionSelector
from .typing import RLoadT

ArrayLike = NDArray | Sequence


logger = logging.getLogger(__name__)


class MeshBuilder:
    def __init__(self, nodes: Sequence[Sequence[int | float]], elements: list[list[int]]) -> None:
        self.assembled = False
        connected: set[int] = set([n for row in elements for n in row[1:]])
        allnodes: set[int] = set([int(row[0]) for row in nodes])
        if disconnected := allnodes.difference(connected):
            for n in disconnected:
                logger.error(f"Node {n} is not connected to any element")
            raise RuntimeError("Disconnected nodes detected")

        self.node_map: collections.Map = collections.Map([int(node[0]) for node in nodes])
        self.elem_map: collections.Map = collections.Map([int(elem[0]) for elem in elements])

        num_node: int = len(nodes)
        max_dim: int = max(len(n[1:]) for n in nodes)
        self.nodes: list[collections.Node] = []
        self.coords: NDArray = np.zeros((num_node, max_dim), dtype=float)
        for i, node in enumerate(nodes):
            xc = [float(x) for x in node[1:]]
            self.coords[i, : len(xc)] = xc
            ni = collections.Node(lid=i, gid=int(node[0]), x=xc)
            self.nodes.append(ni)

        num_elem: int = len(elements)
        max_elem: int = max(len(e[1:]) for e in elements)
        self.connect: NDArray = -np.ones((num_elem, max_elem), dtype=int)
        errors: int = 0
        for i, element in enumerate(elements):
            for j, gid in enumerate(element[1:]):
                if gid not in self.node_map:
                    errors += 1
                    logger.error(f"Node {j + 1} of element {i + 1} ({gid}) is not defined")
                    continue
                self.connect[i, j] = self.node_map.local(gid)
        if errors:
            raise ValueError("Stopping due to previous errors")

        # Meta data to store information needed for one pass mesh assembly
        self.metadata: dict[str, dict] = defaultdict(dict)

        self.edges: list[collections.Edge] = []
        self.blocks: list[TopoBlock] = []
        self.block_elem_map: dict[int, int] = {}
        self.elemsets: dict[str, list[int]] = defaultdict(list)
        self.nodesets: dict[str, list[int]] = defaultdict(list)
        self.sidesets: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def block(self, *, name: str, cell_type: Type[Cell], region: RegionSelector) -> None:
        blocks = self.metadata["blocks"]
        if name in blocks:
            raise ValueError(f"Topo block {name!r} already defined")
        blocks[name] = collections.BlockSpec(name=name, cell_type=cell_type, region=region)  # type: ignore

    def construct_sets(self) -> None:
        self.construct_nodesets()
        self.construct_elemsets()
        self.construct_sidesets()

    def build(self) -> Mesh:
        if self.assembled:
            raise ValueError("MeshBuilder is already assembled")
        self.assemble_blocks()
        self.detect_topology()
        self.assembled = True
        self.construct_sets()
        return Mesh(
            coords=self.coords,
            connect=self.connect,
            blocks=self.blocks,
            node_map=self.node_map,
            elem_map=self.elem_map,
            block_elem_map=self.block_elem_map,
            nodesets=self.nodesets,
            sidesets=self.sidesets,
            elemsets=self.elemsets,
        )

    def assemble_blocks(self) -> None:
        self.blocks.clear()
        self.block_elem_map.clear()
        assigned: set[int] = set()
        for name, spec in self.metadata.get("blocks", {}).items():
            # eids is the global elem index
            eids: list[int] = []
            for e, conn in enumerate(self.connect):
                p = self.coords[conn]
                x = p.mean(axis=0)
                if spec.region(x, on_boundary=False):
                    eids.append(e)

            # By this point, eids holds the local element indices of each element in the block
            mask = np.isin(eids, list(assigned))
            if np.any(mask):
                duplicates = ", ".join(str(eids[i]) for i, m in enumerate(mask) if m)
                raise ValueError(
                    f"Block {name}: attempting to assign elements {duplicates} "
                    "which are already assigned to other topo blocks"
                )
            assigned.update(eids)

            b = len(self.blocks)
            self.block_elem_map.update({eid: b for eid in eids})

            nids: set[int] = set()
            elements: list[list[int]] = []
            for eid in eids:
                nids.update(self.connect[eid])
                elem = [self.elem_map[eid]] + [self.node_map[n] for n in self.connect[eid]]
                elements.append(elem)

            nodes: list[list[int | float]] = []
            for nid in sorted(nids):
                node = [self.node_map[nid]] + self.coords[nid].tolist()
                nodes.append(node)

            block = TopoBlock(name, nodes, elements, spec.cell_type)
            self.blocks.append(block)

        # Check if all elements are assigned to a topo block
        num_elements = self.connect.shape[0]
        if unassigned := set(range(num_elements)).difference(assigned):
            s = ", ".join(str(_) for _ in unassigned)
            raise ValueError(f"Elements {s} not assigned to any element blocks")

    def detect_topology(self) -> None:
        """Detect boundary faces/edges for all blocks and elements."""

        # mapping from face (tuple of sorted node indices) -> list of (block no, local element no, local face no)
        edges: dict[tuple[int, ...], list[tuple[int, int, int]]] = defaultdict(list)

        # Step 1: iterate all blocks and all elements in each block
        for b, block in enumerate(self.blocks):
            for e, conn in enumerate(block.connect):
                for edge_no in range(block.cell_type.nedge):
                    ix = block.cell_type.edge_nodes(edge_no)
                    gids = tuple(sorted([block.node_map[_] for _ in conn[ix]]))
                    edges[gids].append((b, e, edge_no))

        # Step 2: identify faces that are only in one element → boundary
        self.edges.clear()
        edge_normals: dict[int, list[NDArray]] = defaultdict(list)
        for specs in edges.values():
            if len(specs) == 1:
                b, e, edge_no = specs[0]
                block = self.blocks[b]
                conn = block.connect[e]
                p = block.coords[conn]
                normal = block.cell_type.edge_normal(edge_no, p)
                gid = block.elem_map[e]
                lid = self.elem_map.local(gid)
                xd = block.cell_type.edge_centroid(edge_no, p)
                info = collections.Edge(
                    element=lid, x=xd.tolist(), edge=edge_no, normal=normal.tolist()
                )
                self.edges.append(info)
                for ln in block.cell_type.edge_nodes(edge_no):
                    gid = block.node_map[conn[ln]]
                    lid = self.node_map.local(gid)
                    edge_normals[lid].append(normal)

        for lid, normals in edge_normals.items():
            avg_normal = np.mean(normals, axis=0)
            node = self.nodes[lid]
            assert node.lid == lid
            assert node.gid == self.node_map[lid]
            node.normal = avg_normal.tolist()
            node.on_boundary = True
        return

    def nodeset(
        self, name: str, region: RegionSelector | None = None, nodes: list[int] | None = None
    ) -> None:
        if region is None and nodes is None:
            raise ValueError("Expected region or nodes")
        elif region is not None and nodes is not None:
            raise ValueError("Expected region or nodes, not both")
        nodesets = self.metadata["nodesets"]
        if name in [ns[0] for ns in nodesets.values()]:
            raise ValueError(f"Duplicate node set {name!r}")
        nodesets[f"nodeset-{len(nodesets)}"] = (name, region, nodes)

    def construct_nodesets(self) -> None:
        if not self.assembled:
            raise ValueError("Assemble builder before adding constructing node sets")
        self.nodesets.clear()
        name: str
        region: RegionSelector | None
        nodes: list[int] | None
        for name, region, nodes in self.metadata.get("nodesets", {}).values():
            if region is not None:
                for node in self.nodes:
                    if region(node.x, on_boundary=node.on_boundary):  # type: ignore
                        self.nodesets[name].append(node.lid)
                if name not in self.nodesets:
                    raise ValueError(f"{name}: could not find nodes in region")
            elif nodes is not None:
                for gid in nodes:
                    self.nodesets[name].append(self.node_map.local(gid))

    def elemset(self, name: str, region: RegionSelector) -> None:
        elemsets = self.metadata["elemsets"]
        if name in elemsets:
            raise ValueError(f"Duplicate element set {name!r}")
        elemsets[name] = region

    def construct_elemsets(self) -> None:
        if not self.assembled:
            raise ValueError("Assemble builder before adding constructing element sets")
        self.elemsets.clear()
        name: str
        region: RegionSelector
        for name, region in self.metadata.get("elemsets", {}).items():
            for e, conn in enumerate(self.connect):
                p = self.coords[conn]
                x = p.mean(axis=0)
                if region(x, on_boundary=False):  # type: ignore
                    self.elemsets[name].append(e)

    def sideset(self, name: str, region: RegionSelector) -> None:
        sidesets = self.metadata["sidesets"]
        if name in sidesets:
            raise ValueError(f"Duplicate side set {name!r}")
        sidesets[name] = region

    def construct_sidesets(self) -> None:
        if not self.assembled:
            raise ValueError("Assemble builder before adding constructing element sets")
        self.sidesets.clear()
        name: str
        region: RegionSelector
        for name, region in self.metadata.get("sidesets", {}).items():
            for edge in self.edges:
                if region(edge.x, on_boundary=True):  # type: ignore
                    self.sidesets[name].append((edge.element, edge.edge))


class StepBuilder(ABC):
    def __init__(self, name: str, period: float = 1.0) -> None:
        self.name = name
        self.period = period
        self.metadata: dict[str, dict] = defaultdict(dict)

    @abstractmethod
    def build(self, model: Model, parent: Step | None) -> Step: ...


class StaticStepBuilder(StepBuilder):
    def __init__(self, name: str, period: float = 1.0, **options: Any) -> None:
        super().__init__(name=name, period=period)
        self.dbcs: list[tuple[int, float]] = []
        self.nbcs: list[tuple[int, float]] = []
        self.dsloads: DSLoadT = defaultdict(lambda: defaultdict(list))
        self.rloads: RLoadT = defaultdict(lambda: defaultdict(list))
        self.dloads: DLoadT = defaultdict(lambda: defaultdict(list))
        self.mpcs: list[list[int | float]] = []
        self.solver_opts = options

    def boundary(self, *, nodeset: str, dofs: int | list[int], value: float = 0.0) -> None:
        if isinstance(dofs, int):
            dofs = [dofs]
        dbcs = self.metadata["dbcs"]
        dbcs[f"dbc-{len(dbcs)}"] = (nodeset, dofs, value)

    def point_load(self, *, nodeset: str, dofs: int | list[int], value: float = 0.0) -> None:
        if isinstance(dofs, int):
            dofs = [dofs]
        nbcs = self.metadata["nbcs"]
        nbcs[f"nbc-{len(nbcs)}"] = (nodeset, dofs, value)

    def traction(self, *, sideset: str, magnitude: float, direction: Sequence[float]) -> None:
        dsloads = self.metadata["dsloads"]
        dir = normalize(direction)
        dsloads[f"dsload-{len(dsloads)}"] = ("traction", sideset, magnitude, dir)

    def pressure(self, *, sideset: str, magnitude: float) -> None:
        dsloads = self.metadata["dsloads"]
        dsloads[f"dsload-{len(dsloads)}"] = ("pressure", sideset, magnitude)

    def gravity(self, *, elemset: str, g: float, direction: Sequence[float]) -> None:
        dloads = self.metadata["dloads"]
        dir = normalize(direction)
        dloads[f"dload-{len(dloads)}"] = ("gravity", elemset, g, dir)

    def dload(self, *, elemset: str, field: Field) -> None:
        dloads = self.metadata["dloads"]
        dloads[f"dload-{len(dloads)}"] = ("dload", elemset, field)

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

    def build(self, model: Model, parent: Step | None) -> StaticStep:
        self.construct_dbcs(model)
        self.construct_nbcs(model)
        self.construct_dloads(model)
        self.construct_dsloads(model)
        self.construct_rloads(model)
        self.construct_constraints(model)
        return StaticStep(
            name=self.name,
            parent=parent,
            period=self.period,
            dbcs=self.dbcs,
            nbcs=self.nbcs,
            dsloads=self.dsloads,
            dloads=self.dloads,
            rloads=self.rloads,
            equations=self.mpcs,
            solver_options=self.solver_opts,
        )

    def construct_dbcs(self, model: Model) -> None:
        seen: dict[int, float] = {}
        for nodeset, dofs, value in self.metadata.get("dbcs", {}).values():
            if nodeset not in model.nodesets:
                raise ValueError(f"nodeset {nodeset} not defined")
            lids: list[int] = model.nodesets[nodeset]
            for lid in lids:
                for dof in dofs:
                    DOF = model.dof_map[lid, dof]
                    seen[DOF] = value
        self.dbcs = [(k, seen[k]) for k in sorted(seen)]

    def construct_nbcs(self, model: Model) -> None:
        seen: dict[int, float] = defaultdict(float)
        for nodeset, dofs, value in self.metadata.get("nbcs", {}).values():
            if nodeset not in model.nodesets:
                raise ValueError(f"nodeset {nodeset} not defined")
            lids: list[int] = model.nodesets[nodeset]
            for lid in lids:
                for dof in dofs:
                    DOF = model.dof_map[lid, dof]
                    seen[DOF] += value
        self.nbcs = [(k, seen[k]) for k in sorted(seen)]

    def construct_dloads(self, model: Model) -> None:
        dload: DistributedLoad
        for ltype, elemset, *args in self.metadata.get("dloads", {}).values():
            if elemset not in model.elemsets:
                raise ValueError(f"element set {elemset} not defined")
            if ltype == "gravity":
                pass
            elif ltype == "dload":
                field = args[0]
                dload = DistributedLoad(field=field)
            else:
                raise ValueError(f"Unknown ltype: {ltype}")
            for lid in model.elemsets[elemset]:
                gid = model.elem_map[lid]
                block_no = model.block_elem_map[lid]
                block = model.blocks[block_no]
                if ltype == "gravity":
                    g, direction = args
                    dload = GravityLoad(block.material.density * g, direction)
                e = block.elem_map.local(gid)
                self.dloads[block_no][e].append(dload)

    def construct_dsloads(self, model: Model) -> None:
        dsload: DistributedSurfaceLoad
        for ltype, sideset, *args in self.metadata.get("dsloads", {}).values():
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
                self.dsloads[block_no][lid].append((edge_no, dsload))

    def construct_rloads(self, model: Model) -> None:
        sideset: str
        H: NDArray
        u0: NDArray
        for sideset, H, u0 in self.metadata.get("rloads", {}).values():
            if sideset not in model.sidesets:
                raise ValueError(f"side set {sideset} not defined")
            for ele_no, edge_no in model.sidesets[sideset]:
                block_no = model.block_elem_map[ele_no]
                block = model.mesh.blocks[block_no]
                gid = block.elem_map[ele_no]
                lid = block.elem_map.local(gid)
                rload = RobinLoad(edge=edge_no, H=H, u0=u0)
                self.rloads[block_no][lid].append(rload)

    def construct_constraints(self, model: Model) -> None:
        nodes: list[int]
        dofs: list[int]
        coeffs: list[float]
        rhs: float = 0.0
        for nodes, dofs, coeffs, rhs in self.metadata.get("constraints", {}).values():
            mpc: list[int | float] = []
            for gid, dof, coeff in zip(nodes, dofs, coeffs):
                if gid not in model.node_map:
                    raise ValueError(f"Node {gid} is not defined")
                lid = model.node_map.local(gid)
                DOF = model.dof_map[lid, dof]
                mpc.extend([DOF, coeff])
            mpc.append(rhs)
            self.mpcs.append(mpc)


class DirectStepBuilder(StaticStepBuilder):
    def __init__(self, name: str, period: float = 1.0) -> None:
        super().__init__(name, period=period)

    def build(self, model: Model, parent: Step | None) -> DirectStep:  # type: ignore
        self.construct_dbcs(model)
        self.construct_nbcs(model)
        self.construct_dloads(model)
        self.construct_dsloads(model)
        self.construct_rloads(model)
        self.construct_constraints(model)
        return DirectStep(
            name=self.name,
            parent=parent,
            period=self.period,
            dbcs=self.dbcs,
            nbcs=self.nbcs,
            dsloads=self.dsloads,
            dloads=self.dloads,
            rloads=self.rloads,
            equations=self.mpcs,
        )


class HeatTransferStepBuilder(StepBuilder):
    def __init__(self, name: str, period: float = 1.0) -> None:
        super().__init__(name=name, period=period)
        self.dbcs: list[tuple[int, float]] = []
        self.nbcs: list[tuple[int, float]] = []
        self.dsloads: QLoadT = defaultdict(lambda: defaultdict(list))
        self.rloads: RLoadT = defaultdict(lambda: defaultdict(list))
        self.dloads: HLoadT = defaultdict(lambda: defaultdict(list))
        self.mpcs: list[list[int | float]] = []

    def temperature(self, *, nodes: str | int | list[int], value: float = 0.0) -> None:
        dofs = [0]
        dbcs = self.metadata["dbcs"]
        dbcs[f"dbc-{len(dbcs)}"] = (nodes, dofs, value)

    def dflux(self, *, sideset: str, magnitude: float, direction: Sequence[float]) -> None:
        dsloads = self.metadata["dsloads"]
        dir = normalize(direction)
        dsloads[f"dsload-{len(dsloads)}"] = ("flux", sideset, magnitude, dir)

    def source(self, *, elements: str | int | list[int], field: Field) -> None:
        dloads = self.metadata["dloads"]
        dloads[f"dload-{len(dloads)}"] = ("dload", elements, field)

    def film(self, *, sideset: str, h: float, ambient_temp: float) -> None:
        rloads = self.metadata["rloads"]
        rloads[f"rload-{len(rloads)}"] = (sideset, h, ambient_temp)

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

    def build(self, model: Model, parent: Step | None) -> StaticStep:
        self.construct_dbcs(model)
        self.construct_nbcs(model)
        self.construct_dloads(model)
        self.construct_dsloads(model)
        self.construct_rloads(model)
        self.construct_constraints(model)
        return HeatTransferStep(
            name=self.name,
            parent=parent,
            period=self.period,
            dbcs=self.dbcs,
            nbcs=self.nbcs,
            dsloads=self.dsloads,
            dloads=self.dloads,
            rloads=self.rloads,
            equations=self.mpcs,
        )

    def construct_dbcs(self, model: Model) -> None:
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
                    DOF = model.dof_map[lid, dof]
                    seen[DOF] = value
        self.dbcs = [(k, seen[k]) for k in sorted(seen)]

    def construct_nbcs(self, model: Model) -> None:
        seen: dict[int, float] = defaultdict(float)
        for nodeset, dofs, value in self.metadata.get("nbcs", {}).values():
            if nodeset not in model.nodesets:
                raise ValueError(f"nodeset {nodeset} not defined")
            lids: list[int] = model.nodesets[nodeset]
            for lid in lids:
                for dof in dofs:
                    DOF = model.dof_map[lid, dof]
                    seen[DOF] += value
        self.nbcs = [(k, seen[k]) for k in sorted(seen)]

    def construct_dloads(self, model: Model) -> None:
        dload: DistributedLoad
        for ltype, elements, *args in self.metadata.get("dloads", {}).values():
            lids: list[int]
            if isinstance(elements, str):
                if elements not in model.elemsets:
                    raise ValueError(f"element set {elements} not defined")
                lids = model.elemsets[elements]
            elif isinstance(elements, int):
                lids = model.elem_map.local(elements)
            else:
                lids = [model.elem_map.local(gid) for gid in elements]
            if ltype == "dload":
                field = args[0]
                dload = HeatSource(field=field)
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
                self.dloads[block_no][e].append(dload)

    def construct_dsloads(self, model: Model) -> None:
        dsload: DistributedSurfaceLoad
        for ltype, sideset, *args in self.metadata.get("dsloads", {}).values():
            if sideset not in model.sidesets:
                raise ValueError(f"side set {sideset} not defined")
            if ltype == "flux":
                dsload = HeatFlux(magnitude=args[0], direction=args[1])
            else:
                raise ValueError(f"Unknown ltype: {ltype}")
            for elem_no, edge_no in model.sidesets[sideset]:
                block_no = model.block_elem_map[elem_no]
                block = model.blocks[block_no]
                gid = model.elem_map[elem_no]
                lid = block.elem_map.local(gid)
                self.dsloads[block_no][lid].append((edge_no, dsload))

    def construct_rloads(self, model: Model) -> None:
        sideset: str
        H: float
        u0: float
        for sideset, H, u0 in self.metadata.get("rloads", {}).values():
            if sideset not in model.sidesets:
                raise ValueError(f"side set {sideset} not defined")
            for ele_no, edge_no in model.sidesets[sideset]:
                block_no = model.block_elem_map[ele_no]
                block = model.mesh.blocks[block_no]
                gid = block.elem_map[ele_no]
                lid = block.elem_map.local(gid)
                rload = RobinLoad(edge=edge_no, H=np.array([[H]]), u0=np.array([u0]))
                self.rloads[block_no][lid].append(rload)

    def construct_constraints(self, model: Model) -> None:
        nodes: list[int]
        dofs: list[int]
        coeffs: list[float]
        rhs: float = 0.0
        for nodes, dofs, coeffs, rhs in self.metadata.get("constraints", {}).values():
            mpc: list[int | float] = []
            for gid, dof, coeff in zip(nodes, dofs, coeffs):
                if gid not in model.node_map:
                    raise ValueError(f"Node {gid} is not defined")
                lid = model.node_map.local(gid)
                DOF = model.dof_map[lid, dof]
                mpc.extend([DOF, coeff])
            mpc.append(rhs)
            self.mpcs.append(mpc)


class ModelBuilder:
    def __init__(self, mesh: "Mesh", name: str = "Model") -> None:
        self.name = name
        self.assembled = False

        self.mesh = mesh
        self.blocks: list[ElementBlock] = []

        # dof_map[node, dof] -> global (model) dof
        self.node_freedom_table: NDArray = np.empty((0, 0), dtype=int)
        self.node_freedom_types: list[int] = []
        self.block_dof_map: NDArray = np.empty((0, 0), dtype=int)
        self.dof_map: NDArray = np.empty((0, 0), dtype=int)
        self.num_dof: int = -1

        # Meta data to store information needed for one pass mesh assembly
        self.metadata: dict[str, dict] = defaultdict(dict)

        self.steps: list[Step] = []

    def assign_properties(self, *, block: str, element: Element, material: Material) -> None:
        for blk in self.blocks:
            if blk.name == block:
                raise ValueError(f"Element block {block!r} has already been assigned properties")
        for b in self.mesh.blocks:
            if b.name == block:
                blk = ElementBlock.from_topo_block(b, element, material)
                self.blocks.append(blk)
                break
        else:
            raise ValueError(f"Element block {block!r} is not defined")

    def static_step(
        self, name: str | None = None, period: float = 1.0, **options: Any
    ) -> StaticStepBuilder:
        steps = self.metadata["steps"]
        name = name or f"step-{len(steps)}"
        step = steps[name] = StaticStepBuilder(name=name, period=period, **options)
        return step

    def direct_step(self, name: str | None = None, period: float = 1.0) -> DirectStepBuilder:
        steps = self.metadata["steps"]
        name = name or f"step-{len(steps)}"
        step = steps[name] = DirectStepBuilder(name=name, period=period)
        return step

    def heat_transfer_step(
        self, name: str | None = None, period: float = 1.0
    ) -> HeatTransferStepBuilder:
        steps = self.metadata["steps"]
        name = name or f"step-{len(steps)}"
        step = steps[name] = HeatTransferStepBuilder(name=name, period=period)
        return step

    def build(self) -> Model:
        if self.assembled:
            raise ValueError("ModelBuilder already assembled")
        blocks = {block.name for block in self.mesh.blocks}
        if missing := blocks.difference({block.name for block in self.blocks}):
            raise ValueError(f"The following blocks have not been assigned properties: {missing}")
        self.build_dof_maps()
        model = Model(
            name=self.name,
            mesh=self.mesh,
            blocks=self.blocks,
            num_dof=self.num_dof,
            node_freedom_table=self.node_freedom_table,
            node_freedom_types=self.node_freedom_types,
            dof_map=self.dof_map,
            block_dof_map=self.block_dof_map,
        )
        b: StepBuilder
        parent: Step | None = None
        for b in self.metadata.get("steps", {}).values():
            step = b.build(model, parent=parent)
            parent = step
            model.add_step(step)
        self.assembled = True
        return model

    def build_node_freedom_table(self) -> None:
        """
        Build the model-level node freedom table.

        Produces:
            self.node_freedom_table : ndarray[int] of shape (nnode, n_active_dofs)
                1 if the DOF is active at the node, 0 otherwise
            self.node_freedom_types : ndarray[int] of length n_active_dofs
                Physical DOF type corresponding to each column (Ux, Uy, ..., T)
            self.num_dof : int
                Total number of active DOFs in the model
        """

        # -----------------------------
        # 1) Determine max DOF index used across all blocks
        # -----------------------------
        max_dof_idx = -1
        for block in self.blocks:
            for node_dofs in block.element.node_freedom_table:
                max_dof_idx = max(max_dof_idx, max(node_dofs))
        ncol = max_dof_idx + 1  # number of physical DOF types used

        nnode = self.mesh.coords.shape[0]

        # -----------------------------
        # 2) Build full node signature (nnode x ncol)
        # -----------------------------
        node_sig_full = np.zeros((nnode, ncol), dtype=int)

        for block in self.blocks:
            for elem_nodes in block.connect:
                for i, block_node in enumerate(elem_nodes):
                    gid = block.node_map[block_node]
                    lid = self.mesh.node_map.local(gid)
                    for col in block.element.node_freedom_table[i]:
                        node_sig_full[lid, col] = 1

        # -----------------------------
        # 3) Identify active columns
        # -----------------------------
        active_cols = np.where(node_sig_full.sum(axis=0) > 0)[0]

        # -----------------------------
        # 4) Compress node signature and store
        # -----------------------------
        self.node_freedom_table = node_sig_full[:, active_cols].copy()
        self.node_freedom_types = active_cols.tolist()
        self.num_dof = int(self.node_freedom_table.sum())

    def build_dof_map(self) -> None:
        """
        Build global DOF numbering for the model.

        Produces:
            self.dof_map: ndarray[int] of shape (nnode, n_active_dofs)
                Maps (node, local DOF index in node_freedom_table) -> global DOF index
        """
        nnode, n_active = self.node_freedom_table.shape
        self.dof_map = -np.ones((nnode, n_active), dtype=int)

        # Flatten node_freedom_table
        flat_mask = self.node_freedom_table.ravel()
        global_dofs = np.arange(flat_mask.sum(), dtype=int)

        # Assign global DOF indices where DOFs are active
        self.dof_map[self.node_freedom_table == 1] = global_dofs

    def build_block_dof_map(self) -> None:
        """
        Build a precomputed block DOF map:
            block_dof_map[blockno, local_dof] = global DOF label

        After this, `block_freedom_table(blockno)` is simply:
            bft = self.block_dof_map[blockno, :n_block_dof]
        """
        # Determine max DOFs per block
        max_block_dofs = max(b.num_dof for b in self.blocks)

        # Initialize table with -1
        self.block_dof_map = -np.ones((len(self.blocks), max_block_dofs), dtype=int)

        for blockno, block in enumerate(self.blocks):
            local_dof_idx = 0
            for i in range(block.num_nodes):
                gid = block.node_map[i]
                lid = self.mesh.node_map.local(gid)
                for dof_type in block.element.node_freedom_table[0]:
                    col = self.node_freedom_types.index(dof_type)
                    gdof = self.dof_map[lid, col]
                    self.block_dof_map[blockno, local_dof_idx] = gdof
                    local_dof_idx += 1

    def build_dof_maps(self) -> None:
        """
        Build all DOF-related tables for the model.

        Steps:
            1) Build node_freedom_table and node_freedom_types
            2) Build global DOF map: dof_map[node, local_dof]
            3) Build per-block DOF map: block_dof_map[blockno, local_dof]
        """
        # 1) Node-level DOFs
        self.build_node_freedom_table()

        # 2) Global DOF numbering
        self.build_dof_map()

        # 3) Precompute block → global DOFs
        self.build_block_dof_map()


def normalize(a: Sequence[float]) -> NDArray:
    v = np.asarray(a)
    return v / np.linalg.norm(v)
