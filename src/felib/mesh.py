import logging
from collections import defaultdict
from typing import Sequence
from typing import Type

import numpy as np
from numpy.typing import NDArray

from . import collections
from .block import TopoBlock
from .collections import Map
from .element import ReferenceElement
from .pytools import _require_unfrozen
from .pytools import frozen_property
from .typing import RegionSelector

logger = logging.getLogger(__name__)


class Mesh:
    def __init__(self, nodes: Sequence[Sequence[int | float]], elements: list[list[int]]) -> None:
        self.coords: NDArray
        self.connect: NDArray
        self.node_map: Map
        self.elem_map: Map
        self.nodes: list[collections.Node]

        self._init(nodes, elements)

        self._blocks: list[TopoBlock] = []
        self._edges: list[collections.Edge] = []
        self._block_elem_map: dict[int, int] = {}
        self._elemsets: dict[str, list[int]] = defaultdict(list)
        self._nodesets: dict[str, list[int]] = defaultdict(list)
        self._sidesets: dict[str, list[tuple[int, int]]] = defaultdict(list)

        self._frozen = False
        self._builder = _MeshBuilder(self)

    def freeze(self):
        if not self._frozen:
            self._builder.build()
            self._frozen = True

    @_require_unfrozen
    def block(
        self,
        *,
        name: str,
        cell_type: Type[ReferenceElement],
        region: RegionSelector | None = None,
        elements: Sequence[int] | None = None,
    ) -> None:
        return self._builder.block(name=name, cell_type=cell_type, region=region, elements=elements)

    @_require_unfrozen
    def nodeset(
        self, name: str, region: RegionSelector | None = None, nodes: list[int] | None = None
    ) -> None:
        return self._builder.nodeset(name, region=region, nodes=nodes)

    @_require_unfrozen
    def elemset(
        self, name: str, region: RegionSelector | None = None, elements: Sequence[int] | None = None
    ) -> None:
        return self._builder.elemset(name, region=region, elements=elements)

    @_require_unfrozen
    def sideset(
        self,
        name: str,
        region: RegionSelector | None = None,
        sides: list[Sequence[int]] | None = None,
    ) -> None:
        return self._builder.sideset(name, region=region, sides=sides)

    def _init(self, nodes: Sequence[Sequence[int | float]], elements: list[list[int]]) -> None:
        connected: set[int] = set([n for row in elements for n in row[1:]])
        allnodes: set[int] = set([int(row[0]) for row in nodes])
        if disconnected := allnodes.difference(connected):
            for n in disconnected:
                logger.error(f"Node {n} is not connected to any element")
            raise RuntimeError("Disconnected nodes detected")

        self.node_map = collections.Map([int(node[0]) for node in nodes])
        self.elem_map = collections.Map([int(elem[0]) for elem in elements])

        nnode: int = len(nodes)
        max_dim: int = max(len(n[1:]) for n in nodes)
        self.coords = np.zeros((nnode, max_dim), dtype=float)
        self.nodes = []
        for i, node in enumerate(nodes):
            xc = [float(x) for x in node[1:]]
            self.coords[i, : len(xc)] = xc
            ni = collections.Node(lid=i, gid=int(node[0]), x=xc)
            self.nodes.append(ni)

        nelem: int = len(elements)
        max_elem: int = max(len(e[1:]) for e in elements)
        self.connect = -np.ones((nelem, max_elem), dtype=int)
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

    @frozen_property
    def blocks(self) -> list[TopoBlock]:
        return self._blocks

    @frozen_property
    def edges(self) -> list[collections.Edge]:
        return self._edges

    @frozen_property
    def block_elem_map(self) -> dict[int, int]:
        return self._block_elem_map

    @frozen_property
    def elemsets(self) -> dict[str, list[int]]:
        return self._elemsets

    @frozen_property
    def nodesets(self) -> dict[str, list[int]]:
        return self._nodesets

    @frozen_property
    def sidesets(self) -> dict[str, list[tuple[int, int]]]:
        return self._sidesets


class _MeshBuilder:
    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh
        self.assembled = False
        # Meta data to store information needed for one pass mesh assembly
        self.metadata: dict[str, dict] = defaultdict(dict)

    def block(
        self,
        *,
        name: str,
        cell_type: Type[ReferenceElement],
        region: RegionSelector | None = None,
        elements: Sequence[int] | None = None,
    ) -> None:
        if region is None and elements is None:
            raise ValueError("Expected region or elements")
        elif region is not None and elements is not None:
            raise ValueError("Expected region or elements, not both")
        blocks = self.metadata["blocks"]
        if name in blocks:
            raise ValueError(f"Topo block {name!r} already defined")
        blocks[name] = collections.BlockSpec(
            name=name,
            cell_type=cell_type,
            elements=elements,
            region=region,  # type: ignore
        )

    def construct_sets(self) -> None:
        self.construct_nodesets()
        self.construct_elemsets()
        self.construct_sidesets()

    def build(self) -> None:
        if self.assembled:
            raise ValueError("MeshBuilder.build() already called")
        self.assemble_blocks()
        self.detect_topology()
        self.assembled = True
        self.construct_sets()

    def assemble_blocks(self) -> None:
        mesh = self.mesh
        mesh._blocks.clear()
        mesh._block_elem_map.clear()
        assigned: set[int] = set()
        for name, spec in self.metadata.get("blocks", {}).items():
            # eids is the global elem index
            eids: list[int] = []
            if spec.elements is not None:
                eids.extend([mesh.elem_map.local(el) for el in spec.elements])
            else:
                assert callable(spec.region)
                for e, conn in enumerate(mesh.connect):
                    p = mesh.coords[conn]
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

            b = len(mesh._blocks)
            mesh._block_elem_map.update({eid: b for eid in eids})

            nids: set[int] = set()
            elements: list[list[int]] = []
            for eid in eids:
                nids.update(mesh.connect[eid])
                elem = [mesh.elem_map[eid]] + [mesh.node_map[n] for n in mesh.connect[eid]]
                elements.append(elem)

            nodes: list[list[int | float]] = []
            for nid in sorted(nids):
                node = [mesh.node_map[nid]] + mesh.coords[nid].tolist()
                nodes.append(node)

            block = TopoBlock(name, nodes, elements, spec.cell_type)
            mesh._blocks.append(block)

        # Check if all elements are assigned to a topo block
        nelem = mesh.connect.shape[0]
        if unassigned := set(range(nelem)).difference(assigned):
            s = ", ".join(str(_) for _ in unassigned)
            raise ValueError(f"Elements {s} not assigned to any element blocks")

    def detect_topology(self) -> None:
        """Detect boundary faces/edges for all blocks and elements."""

        # mapping from face (tuple of sorted node indices) -> list of (block no, local element no, local face no)
        edges: dict[tuple[int, ...], list[tuple[int, int, int]]] = defaultdict(list)

        # Step 1: iterate all blocks and all elements in each block
        for b, block in enumerate(self.mesh._blocks):
            for e, conn in enumerate(block.connect):
                for edge_no in range(block.ref_el.nedge):
                    ix = block.ref_el.edge_nodes(edge_no)
                    gids = tuple(sorted([block.node_map[_] for _ in conn[ix]]))
                    edges[gids].append((b, e, edge_no))

        # Step 2: identify faces that are only in one element → boundary
        self.mesh._edges.clear()
        edge_normals: dict[int, list[NDArray]] = defaultdict(list)
        for specs in edges.values():
            if len(specs) == 1:
                b, e, edge_no = specs[0]
                block = self.mesh._blocks[b]
                conn = block.connect[e]
                p = block.coords[conn]
                normal = block.ref_el.edge_normal(edge_no, p)
                gid = block.elem_map[e]
                lid = self.mesh.elem_map.local(gid)
                xd = block.ref_el.edge_centroid(edge_no, p)
                info = collections.Edge(
                    element=lid, x=xd.tolist(), edge=edge_no, normal=normal.tolist()
                )
                self.mesh._edges.append(info)
                for ln in block.ref_el.edge_nodes(edge_no):
                    gid = block.node_map[conn[ln]]
                    lid = self.mesh.node_map.local(gid)
                    edge_normals[lid].append(normal)

        for lid, normals in edge_normals.items():
            avg_normal = np.mean(normals, axis=0)
            node = self.mesh.nodes[lid]
            assert node.lid == lid
            assert node.gid == self.mesh.node_map[lid]
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
        self.mesh._nodesets.clear()
        name: str
        region: RegionSelector | None
        nodes: list[int] | None
        for name, region, nodes in self.metadata.get("nodesets", {}).values():
            if region is not None:
                for node in self.mesh.nodes:
                    if region(node.x, on_boundary=node.on_boundary):  # type: ignore
                        self.mesh._nodesets[name].append(node.lid)
            elif nodes is not None:
                for gid in nodes:
                    self.mesh._nodesets[name].append(self.mesh.node_map.local(gid))
            else:
                raise ValueError("Cannot construct nodeset from empty nodeset data")
            if name not in self.mesh._nodesets:
                raise ValueError(f"{name}: could not find nodes in region")

    def elemset(
        self, name: str, region: RegionSelector | None = None, elements: Sequence[int] | None = None
    ) -> None:
        if region is None and elements is None:
            raise ValueError("Expected region or elements")
        elif region is not None and elements is not None:
            raise ValueError("Expected region or elements, not both")
        elemsets = self.metadata["elemsets"]
        if name in elemsets:
            raise ValueError(f"Duplicate element set {name!r}")
        elemsets[f"elemset-{len(elemsets)}"] = (name, region, elements)

    def construct_elemsets(self) -> None:
        self.mesh._elemsets.clear()
        name: str
        region: RegionSelector | None
        elements: Sequence[int] | None
        for name, region, elements in self.metadata.get("elemsets", {}).values():
            if region is not None:
                for e, conn in enumerate(self.mesh.connect):
                    p = self.mesh.coords[conn]
                    x = p.mean(axis=0)
                    if region(x, on_boundary=False):  # type: ignore
                        self.mesh._elemsets[name].append(e)
            elif elements is not None:
                for el in elements:
                    self.mesh._elemsets[name].append(self.mesh.elem_map.local(el))

    def sideset(
        self, name: str, region: RegionSelector | None, sides: list[Sequence[int]] | None = None
    ) -> None:
        if region is None and sides is None:
            raise ValueError("Expected region or sides")
        elif region is not None and sides is not None:
            raise ValueError("Expected region or sides, not both")
        sidesets = self.metadata["sidesets"]
        if name in sidesets:
            raise ValueError(f"Duplicate side set {name!r}")
        sidesets[f"sideset-{len(sidesets)}"] = (name, region, sides)

    def construct_sidesets(self) -> None:
        if not self.assembled:
            raise ValueError("Assemble builder before constructing element sets")
        self.mesh._sidesets.clear()
        name: str
        region: RegionSelector | None
        sides: list[Sequence[int]] | None
        for name, region, sides in self.metadata.get("sidesets", {}).values():
            if region is not None:
                for edge in self.mesh._edges:
                    if region(edge.x, on_boundary=True):  # type: ignore
                        self.mesh._sidesets[name].append((edge.element, edge.edge))
            elif sides is not None:
                for el, side in sides:
                    self.mesh._sidesets[name].append((self.mesh.elem_map.local(el), side - 1))
