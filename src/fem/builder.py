import logging
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from typing import Sequence
from typing import Type

import numpy as np
from numpy.typing import NDArray

from .step import Step
from .step import StaticStep
from . import collections
from .block import ElementBlock
from .block import TopoBlock
from .cell import Cell
from .collections import DistributedLoad
from .collections import DistributedSurfaceLoad
from .collections import PressureLoad
from .collections import TractionLoad
from .collections import RobinLoad
from .collections import ConstantNodalField
from .collections import GravityLoad
from .collections import Field
from .collections import Load
from .collections import NodalFieldEvaluator
from .element import Element
from .material import Material
from .model import Model
from .typing import RegionSelector

ArrayLike = NDArray | Sequence


logger = logging.getLogger(__name__)


class MeshBuilder:
    def __init__(self, nodes: list[Sequence[int | float]], elements: list[list[int]]) -> None:
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

    def assemble(self) -> None:
        if self.assembled:
            raise ValueError("MeshBuilder is already assembled")
        self.assemble_blocks()
        self.detect_topology()
        self.assembled = True
        self.construct_sets()

    def construct_sets(self) -> None:
        self.construct_nodesets()
        self.construct_elemsets()
        self.construct_sidesets()

    def emit_mesh(self) -> "Mesh":
        if not self.assembled:
            self.assemble()
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
                for node in nodes:
                    self.nodesets[name].append(self.node_map.local(node))

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


DSloadT = dict[int, dict[int, list[tuple[int, DistributedSurfaceLoad]]]]
RloadT = dict[int, dict[int, list[RobinLoad]]]
DloadT = dict[int, dict[int, list[DistributedLoad]]]

# {b: {e: evaluator}}

class ModelBuilder:
    def __init__(self, mesh: "Mesh") -> None:
        self.assembled = False

        self.mesh = mesh
        self.blocks: list[ElementBlock] = []

        # dof_map[node, dof] -> global (model) dof
        self.dof_map: NDArray = np.empty((0, 0), dtype=int)
        self.node_signatures: NDArray = np.empty((0, 0), dtype=int)

        # Meta data to store information needed for one pass mesh assembly
        self.metadata: dict[str, dict] = defaultdict(dict)

        self.dbcs: list[tuple[int, int, NodalFieldEvaluator]] = []
        self.nbcs: list[tuple[int, int, NodalFieldEvaluator]] = []
        self.dsloads: DSloadT = defaultdict(lambda: defaultdict(list))
        self.rloads: RloadT = defaultdict(lambda: defaultdict(list))
        self.dloads: DloadT = defaultdict(lambda: defaultdict(list))
        self.mpcs: list[list[int | float]] = []

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

    def boundary(
        self,
        *,
        nodeset: str,
        dofs: int | list[int],
        amplitude: float | NodalFieldEvaluator = 0.0,
    ) -> None:
        if nodeset not in self.mesh.nodesets:
            raise ValueError(f"nodeset {nodeset} not defined")
        lids: list[int] = self.mesh.nodesets[nodeset]
        if isinstance(amplitude, float):
            amplitude = ConstantNodalField(amplitude)
        if isinstance(dofs, int):
            dofs = [dofs]
        for lid in lids:
            for dof in dofs:
                self.dbcs.append((lid, dof, amplitude))

    def static_step(self, name: str | None = None, period: float = 1.0) -> StaticStep:
        parent = None if not self.steps else self.steps[-1]
        name = name or f"Step-{len(self.steps) + 1}"
        return StaticStep(name=name, period=period, parent=parent)

    def point_load(
        self,
        *,
        nodeset: str,
        dofs: int | list[int],
        amplitude: float | NodalFieldEvaluator = 0.0,
    ) -> None:
        if nodeset not in self.mesh.nodesets:
            raise ValueError(f"nodeset {nodeset} not defined")
        lids: list[int] = self.mesh.nodesets[nodeset]
        if isinstance(amplitude, float):
            amplitude = ConstantNodalField(amplitude)
        if isinstance(dofs, int):
            dofs = [dofs]
        for lid in lids:
            for dof in dofs:
                self.nbcs.append((lid, dof, amplitude))

    def traction(self, *, sideset: str, magnitude: float, direction: Sequence[float]) -> None:
        if sideset not in self.mesh.sidesets:
            raise ValueError(f"sideset {sideset} not defined")
        dsloads = self.metadata["dsloads"]
        dsloads[f"dsload-{len(dsloads)}"] = ("traction", sideset, magnitude, direction)

    def pressure(self, *, sideset: str, magnitude: float) -> None:
        if sideset not in self.mesh.sidesets:
            raise ValueError(f"sideset {sideset} not defined")
        dsloads = self.metadata["dsloads"]
        dsloads[f"dsload-{len(dsloads)}"] = ("pressure", sideset, magnitude)

    def robin(self, *, sideset: str, u0: collections.Vector, H: collections.Matrix) -> None:
        if sideset not in self.mesh.sidesets:
            raise ValueError(f"sideset {sideset} not defined")
        rloads = self.metadata["rloads"]
        rloads[f"rload-{len(rloads)}"] = (sideset, H, u0)

    def gravity(self, *, elemset: str, g: float, direction: Sequence[float]
    ) -> None:
        if elemset not in self.mesh.elemsets:
            raise ValueError(f"element set {elemset} not defined")
        dloads = self.metadata["dloads"]
        dloads[f"dload-{len(dloads)}"] = ("gravity", elemset, g, direction)

    def dload(self, *, elemset: str, field: Field) -> None:
        if elemset not in self.mesh.elemsets:
            raise ValueError(f"element set {elemset} not defined")
        dloads = self.metadata["dloads"]
        dloads[f"dload-{len(dloads)}"] = ("dload", elemset, field)

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
        constraints = self.metadata["constraints"]
        constraints[f"constraint-{len(constraints)}"] = (nodes, dofs, coeffs, rhs)

    def construct_dloads(self) -> None:
        if not self.assembled:
            raise ValueError("Assemble builder before constructing distributed loads")
        dload: Load
        for ltype, elemset, *args in self.metadata.get("dloads", {}).values():
            if ltype == "gravity":
                g, direction = args
                dload = GravityLoad(g, direction)
            elif ltype == "dload":
                field = args[0]
                dload = DistributedLoad(field=field)
            else:
                raise ValueError(f"Unknown ltype: {ltype}")
            for lid in self.mesh.elemsets[elemset]:
                gid = self.mesh.elem_map[lid]
                block_no = self.mesh.block_elem_map[lid]
                block = self.mesh.blocks[block_no]
                e = block.elem_map.local(gid)
                self.dloads[block_no][e].append(dload)

    def construct_dsloads(self) -> None:
        if not self.assembled:
            raise ValueError("Assemble builder before constructing surface loads")
        sideset: str
        dsload: DistributedSurfaceLoad
        for ltype, sideset, *args in self.metadata.get("dsloads", {}).values():
            if ltype == "traction":
                magnitude, direction = args
                dsload = TractionLoad(magnitude=magnitude, direction=direction)
            elif ltype == "pressure":
                magnitude = args[0]
                dsload = PressureLoad(magnitude=magnitude)
            else:
                raise ValueError(f"Unknown ltype: {ltype}")
            for ele_no, edge_no in self.mesh.sidesets[sideset]:
                block_no = self.mesh.block_elem_map[ele_no]
                block = self.mesh.blocks[block_no]
                gid = self.mesh.elem_map[ele_no]
                lid = block.elem_map.local(gid)
                self.dsloads[block_no][lid].append((edge_no, dsload))

    def construct_rloads(self) -> None:
        if not self.assembled:
            raise ValueError("Assemble builder before constructing Robin conditions")
        sideset: str
        u0: collections.Vector
        H: collections.Vector
        for sideset, H, u0 in self.metadata.get("rloads", {}).values():
            for ele_no, edge_no in self.mesh.sidesets[sideset]:
                block_no = self.mesh.block_elem_map[ele_no]
                block = self.mesh.blocks[block_no]
                conn = self.mesh.connect[ele_no]
                p = self.mesh.coords[conn]
                x = block.cell_type.edge_centroid(edge_no, p)

                gid = self.mesh.elem_map[ele_no]
                lid = block.elem_map.local(gid)
                rload = RobinLoad(
                    edge=edge_no, H=np.array(H(x.tolist())), u0=np.array(u0(x.tolist()))
                )
                self.rloads[block_no][lid].append(rload)

    def construct_constraints(self) -> None:
        if not self.assembled:
            raise ValueError("Assemble builder before constructing constraints")
        nodes: list[int]
        dofs: list[int]
        coeffs: list[float]
        rhs: float = 0.0
        for nodes, dofs, coeffs, rhs in self.metadata.get("constraints", {}).values():
            mpc: list[int | float] = []
            for gid, dof, coeff in zip(nodes, dofs, coeffs):
                if gid not in self.mesh.node_map:
                    raise ValueError(f"Node {gid} is not defined")
                mpc.extend([self.mesh.node_map.local(gid), dof, coeff])
            mpc.append(rhs)
            self.mpcs.append(mpc)

    def assemble(self) -> None:
        if self.assembled:
            raise ValueError("ModelBuilder already assembled")
        blocks = {block.name for block in self.mesh.blocks}
        if missing := blocks.difference({block.name for block in self.blocks}):
            raise ValueError(f"The following blocks have not been assigned properties: {missing}")
        self.build_dof_maps()
        self.assembled = True
        self.construct_loads()
        self.construct_constraints()

    def construct_loads(self) -> None:
        self.construct_dloads()
        self.construct_dsloads()
        self.construct_rloads()

    def emit_model(self) -> Model:
        if not self.assembled:
            self.assemble()
        return Model(
            dof_map=self.dof_map,
            coords=self.mesh.coords,
            connect=self.mesh.connect,
            blocks=self.blocks,
            num_dof=self.num_dof,
            block_dof_map=self.block_dof_map,
            node_signatures=self.node_signatures,
            node_map=self.mesh.node_map,
            elem_map=self.mesh.elem_map,
            dirichlet_bcs=self.dbcs,
            neumann_bcs=self.nbcs,
            robin_loads=self.rloads,
            dsloads=self.dsloads,
            dloads=self.dloads,
            equations=self.mpcs,
        )

    def build_dof_maps(self) -> None:
        # Build node signatures and dof maps
        num_node: int = self.mesh.coords.shape[0]
        max_dof_per_node = max([b.element.dof_per_node for b in self.blocks])
        # FIX ME: compress out dofs not used at all
        max_dof_per_node = max(max_dof_per_node, 10)
        self.node_signatures = np.zeros((num_node, max_dof_per_node), dtype=int)
        for b, block in enumerate(self.blocks):
            node_freedoms = np.asarray(block.element.node_freedom_table, dtype=int)
            for nodes in block.connect:
                # Map block local node number to model node number
                ix = []
                for node in nodes:
                    gid = block.node_map[node]
                    lid = self.mesh.node_map.local(gid)
                    ix.append(lid)
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
                gid = block.node_map[node]
                lid = self.mesh.node_map.local(gid)
                for local_dof in range(block.element.dof_per_node):
                    dof = block.dof_map[node, local_dof]
                    DOF = self.dof_map[lid, local_dof]
                    self.block_dof_map[b, dof] = DOF


@dataclass
class Mesh:
    coords: NDArray
    connect: NDArray
    blocks: list[TopoBlock]
    node_map: collections.Map
    elem_map: collections.Map
    block_elem_map: dict[int, int]
    nodesets: dict[str, list[int]] = field(default_factory=dict)
    sidesets: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    elemsets: dict[str, list[int]] = field(default_factory=dict)
