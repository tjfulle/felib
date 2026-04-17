import logging
from typing import TYPE_CHECKING
from typing import Sequence
from typing import Type

import numpy as np
from numpy.typing import NDArray

from .collections import DistributedLoad
from .collections import DistributedSurfaceLoad
from .collections import ElementBlockData
from .collections import Map
from .collections import RobinLoad

if TYPE_CHECKING:
    from .element import Element
    from .element import ReferenceElement
    from .material import Material


logger = logging.getLogger(__name__)


class TopoBlock:
    def __init__(
        self,
        name: str,
        nodes: list[list[int | float]],
        elements: list[list[int]],
        cell_type: Type["ReferenceElement"],
    ) -> None:
        self.name = name
        self.ref_el = cell_type()

        connected: set[int] = set([gid for elem in elements for gid in elem[1:]])
        allnodes: set[int] = set([int(node[0]) for node in nodes])
        if disconnected := allnodes.difference(connected):
            for n in disconnected:
                logger.error(f"{self}: node {n} is not connected to any cell")
            raise RuntimeError("Disconnected nodes detected")

        self.node_map = Map([int(node[0]) for node in nodes])
        self.elem_map = Map([int(elem[0]) for elem in elements])

        self.coords: NDArray = np.zeros((len(nodes), self.ref_el.ndim), dtype=float)
        for n, node in enumerate(nodes):
            xc = [float(x) for x in node[1:]]
            if len(xc) != self.ref_el.ndim:
                raise ValueError(
                    f"{self}: node {node[0]} has {len(xc)} dims, expected {self.ref_el.ndim}"
                )
            self.coords[n] = xc

        self.connect = np.zeros((len(elements), self.ref_el.nnode), dtype=int)
        for e, elem in enumerate(elements):
            lids = [self.node_map.local(gid) for gid in elem[1:]]
            if len(lids) != self.ref_el.nnode:
                raise ValueError(
                    f"{self}: element {elem[0]} has {len(lids)} nodes, expected {self.ref_el.nnode}"
                )
            self.connect[e] = lids

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class ElementBlock:
    def __init__(
        self,
        name: str,
        coords: NDArray,
        connect: NDArray,
        element: "Element",
        material: "Material",
        elem_map: Map,
        node_map: Map,
    ) -> None:
        self.name = name
        self.coords = np.asarray(coords)
        self.connect = np.asarray(connect)
        self.element = element
        self.material = material
        self.node_map = node_map
        self.elem_map = elem_map
        self.nnode = self.coords.shape[0]
        self.nelem = self.connect.shape[0]
        self.ndof = self.nnode * self.element.dof_per_node
        # dof_map[node, dof] -> block (global) dof
        self.dof_map = np.arange(self.ndof, dtype=int).reshape(self.nnode, -1)

    @classmethod
    def from_topo_block(
        cls, blk: TopoBlock, element: "Element", material: "Material"
    ) -> "ElementBlock":
        return cls(
            name=blk.name,
            coords=blk.coords,
            connect=blk.connect,
            element=element,
            material=material,
            elem_map=blk.elem_map,
            node_map=blk.node_map,
        )

    @property
    def active_dofs(self) -> tuple[int, ...]:
        nft = self.element.node_freedom_table[0]
        return tuple([int(i) for i, x in enumerate(nft) if x])

    def element_variable_names(self) -> list[str]:
        names = self.element.history_variables()
        names.extend(self.material.history_variables())
        return names

    def assemble(
        self,
        step: int,
        increment: int,
        time: Sequence[float],
        dt: float,
        u: NDArray,
        du: NDArray,
        data: ElementBlockData,
        dloads: dict[int, list[DistributedLoad]] | None = None,
        dsloads: dict[int, list[tuple[int, DistributedSurfaceLoad]]] | None = None,
        rloads: dict[int, list[RobinLoad]] | None = None,
    ) -> tuple[NDArray, NDArray]:
        data.sync()
        K = np.zeros((self.ndof, self.ndof), dtype=float)
        R = np.zeros(self.ndof, dtype=float)
        dloads = dloads or {}
        dsloads = dsloads or {}
        rloads = rloads or {}
        for e, nodes in enumerate(self.connect):
            eleno = self.elem_map[e]
            eft = self.element_freedom_table(nodes)
            ke, re = self.element.eval(
                self.material,
                step,
                increment,
                time,
                dt,
                eleno,
                self.coords[nodes],
                u[eft],
                du[eft],
                data.scratch[e],
                dloads=dloads.get(e),
                dsloads=dsloads.get(e),
                rloads=rloads.get(e),
            )
            K[np.ix_(eft, eft)] += ke
            R[eft] += re
        return K, R

    def element_freedom_table(self, nodes: NDArray) -> list[int]:
        dof_per_node = self.element.dof_per_node
        eft = [dof_per_node * node + j for node in nodes for j in range(dof_per_node)]
        return eft
