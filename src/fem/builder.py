import logging
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import Sequence
from typing import Type

import numpy as np
from numpy.typing import NDArray

from . import collections
from .block import ElementBlock
from .block import TopoBlock
from .cell import Cell
from .collections import DistributedLoad
from .collections import RobinLoad
from .collections import SurfaceLoad
from .constants import GRAVITY
from .constants import PRESSURE
from .constants import TRACTION
from .element import Element
from .material import Material
from .model import Model

ArrayLike = NDArray | Sequence


logger = logging.getLogger(__name__)


RegionSelector = Callable[[Sequence[float], bool], bool]


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

            # By this point, eids holds the global element indices of each element in the block
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

    def nodeset(self, name: str, region: RegionSelector) -> None:
        nodesets = self.metadata["nodesets"]
        if name in nodesets:
            raise ValueError(f"Duplicate node set {name!r}")
        nodesets[name] = region

    def construct_nodesets(self) -> None:
        if not self.assembled:
            raise ValueError("Assemble builder before adding constructing node sets")
        self.nodesets.clear()
        name: str
        region: RegionSelector
        for name, region in self.metadata.get("nodesets", {}).items():
            for node in self.nodes:
                if region(node.x, on_boundary=node.on_boundary):  # type: ignore
                    self.nodesets[name].append(node.lid)

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


SloadT = dict[int, dict[int, list[SurfaceLoad]]]
RloadT = dict[int, dict[int, list[RobinLoad]]]
DloadT = dict[int, dict[int, list[DistributedLoad]]]


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

        self.dbcs: dict[int, float] = defaultdict(float)
        self.nbcs: dict[int, float] = defaultdict(float)
        self.sloads: SloadT = defaultdict(lambda: defaultdict(list))
        self.rloads: RloadT = defaultdict(lambda: defaultdict(list))
        self.dloads: DloadT = defaultdict(lambda: defaultdict(list))
        self.mpcs: list[list[int | float]] = []

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

    def boundary(self, *, nodeset: str, bc: collections.BoundaryCondition) -> None:
        if nodeset not in self.mesh.nodesets:
            raise ValueError(f"nodeset {nodeset} not defined")
        dbcs = self.metadata["dbcs"]
        dbcs[f"dbc-{len(dbcs)}"] = (nodeset, bc)

    def point_load(self, *, nodeset: str, vector: collections.Vector) -> None:
        if nodeset not in self.mesh.nodesets:
            raise ValueError(f"nodeset {nodeset} not defined")
        nbcs = self.metadata["nbcs"]
        nbcs[f"nbc-{len(nbcs)}"] = (nodeset, vector)

    def traction(self, *, sideset: str, vector: collections.Vector) -> None:
        if sideset not in self.mesh.sidesets:
            raise ValueError(f"sideset {sideset} not defined")
        sloads = self.metadata["sloads"]
        sloads[f"sload-{len(sloads)}"] = (sideset, TRACTION, vector)

    def pressure(self, *, sideset: str, vector: collections.Value) -> None:
        if sideset not in self.mesh.sidesets:
            raise ValueError(f"sideset {sideset} not defined")
        sloads = self.metadata["sloads"]
        sloads[f"sload-{len(sloads)}"] = (sideset, PRESSURE, vector)

    def robin(self, *, sideset: str, u0: collections.Vector, H: collections.Matrix) -> None:
        if sideset not in self.mesh.sidesets:
            raise ValueError(f"sideset {sideset} not defined")
        rloads = self.metadata["rloads"]
        rloads[f"rload-{len(rloads)}"] = (sideset, H, u0)

    def gravity(self, *, elemset: str, vector: collections.Vector) -> None:
        if elemset not in self.mesh.elemsets:
            raise ValueError(f"Element set {elemset} not defined")
        dloads = self.metadata["dloads"]
        dloads[f"dload-{len(dloads)}"] = (elemset, GRAVITY, vector)

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

    def construct_dbcs(self) -> None:
        if not self.assembled:
            raise ValueError("Assemble builder before constructing boundary conditions")
        nodeset: str
        bc: collections.BoundaryCondition
        for nodeset, bc in self.metadata.get("dbcs", {}).values():
            lids = self.mesh.nodesets[nodeset]
            for lid in lids:
                x = self.mesh.coords[lid]
                for i, value in bc(x):
                    dof = int(self.dof_map[lid, i])
                    self.dbcs[dof] = float(value)

    def construct_nbcs(self) -> None:
        if not self.assembled:
            raise ValueError("Assemble builder before constructing point loads")
        nodeset: str
        vector: collections.Vector
        for nodeset, vector in self.metadata.get("nbcs", {}).values():
            lids = self.mesh.nodesets[nodeset]
            for lid in lids:
                x = self.mesh.coords[lid]
                for i, vi in enumerate(vector(x)):
                    if abs(vi) <= 0.0:
                        continue
                    dof = int(self.dof_map[lid, i])
                    self.nbcs[dof] += float(vi)

    def construct_dloads(self) -> None:
        if not self.assembled:
            raise ValueError("Assemble builder before constructing distributed loads")
        elemset: str
        load_type: int
        vector: collections.Vector
        for elemset, load_type, vector in self.metadata.get("dloads", {}).values():
            for ele_no in self.mesh.elemsets[elemset]:
                block_no = self.mesh.block_elem_map[ele_no]
                block = self.mesh.blocks[block_no]
                p = self.mesh.coords[self.mesh.connect[ele_no]]
                x = p.mean(axis=0)
                gid = self.mesh.elem_map[ele_no]
                lid = block.elem_map.local(gid)
                dload = DistributedLoad(load_type=load_type, value=np.array(vector(x.tolist())))
                self.dloads[block_no][lid].append(dload)

    def construct_sloads(self) -> None:
        if not self.assembled:
            raise ValueError("Assemble builder before constructing surface loads")
        sideset: str
        load_type: int
        vector: collections.Vector
        for sideset, load_type, vector in self.metadata.get("sloads", {}).values():
            for ele_no, edge_no in self.mesh.sidesets[sideset]:
                block_no = self.mesh.block_elem_map[ele_no]
                block = self.mesh.blocks[block_no]
                conn = self.mesh.connect[ele_no]
                p = self.mesh.coords[conn]
                x = block.cell_type.edge_centroid(edge_no, p)

                gid = self.mesh.elem_map[ele_no]
                lid = block.elem_map.local(gid)
                sload = SurfaceLoad(
                    edge=edge_no, load_type=load_type, value=np.array(vector(x.tolist()))
                )
                self.sloads[block_no][lid].append(sload)

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
        self.construct_bcs()
        self.construct_loads()
        self.construct_constraints()

    def construct_bcs(self) -> None:
        self.construct_dbcs()
        self.construct_nbcs()

    def construct_loads(self) -> None:
        self.construct_dloads()
        self.construct_sloads()
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
            dirichlet_dofs=np.asarray(list(self.dbcs.keys()), dtype=int),
            dirichlet_vals=np.asarray(list(self.dbcs.values()), dtype=float),
            neumann_dofs=np.asarray(list(self.nbcs.keys()), dtype=int),
            neumann_vals=np.asarray(list(self.nbcs.values()), dtype=float),
            robin_loads=self.rloads,
            surface_loads=self.sloads,
            distributed_loads=self.dloads,
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
