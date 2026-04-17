import logging
from typing import Sequence

from numpy.typing import NDArray

from .block import ElementBlock
from .collections import Map
from .element import Element
from .material import Material
from .mesh import Mesh
from .pytools import _require_unfrozen

# Combined type union for array-like input
ArrayLike = NDArray | Sequence


logger = logging.getLogger(__name__)


class Model:
    """Finite element model container class."""

    def __init__(self, mesh: "Mesh", name: str = "Model-1") -> None:
        self.name = name
        self.mesh = mesh
        self.mesh.freeze()
        self._frozen = False
        self._blocks: list[ElementBlock] = []

    def freeze(self) -> None:
        if not self._frozen:
            blocks = {block.name for block in self.mesh._blocks}
            if missing := blocks.difference({block.name for block in self._blocks}):
                raise ValueError(
                    f"The following blocks have not been assigned properties: {missing}"
                )
            self._frozen = True

    @property
    def blocks(self) -> list[ElementBlock]:
        if self._blocks is None:
            raise RuntimeError("Cannot access attribute 'blocks' of unfrozen model")
        return self._blocks

    @property
    def nnode(self) -> int:
        """Return the number of global nodes in the mesh."""
        return self.mesh.coords.shape[0]

    @property
    def ndim(self) -> int:
        return self.mesh.coords.shape[1]

    @property
    def nelem(self) -> int:
        """Return the number of global elements in the mesh."""
        return self.mesh.coords.shape[0]

    @property
    def node_map(self) -> Map:
        """Return the global node mapping object."""
        return self.mesh.node_map

    @property
    def elem_map(self) -> Map:
        """Return the global element mapping object."""
        return self.mesh.elem_map

    @property
    def coords(self) -> NDArray:
        """Return global coordinates of all nodes."""
        return self.mesh.coords

    @property
    def connect(self) -> NDArray:
        """Return element connectivity array."""
        return self.mesh.connect

    @property
    def elemsets(self) -> dict[str, list[int]]:
        """Return element sets defined in the mesh."""
        return self.mesh.elemsets

    @property
    def nodesets(self) -> dict[str, list[int]]:
        """Return node sets defined in the mesh."""
        return self.mesh.nodesets

    @property
    def sidesets(self) -> dict[str, list[tuple[int, int]]]:
        """Return side sets defined in the mesh."""
        return self.mesh.sidesets

    @property
    def block_elem_map(self) -> dict[int, int]:
        """Return mapping from block index to element index."""
        return self.mesh.block_elem_map

    @_require_unfrozen
    def assign_properties(self, *, block: str, element: Element, material: Material) -> None:
        for blk in self._blocks:
            if blk.name == block:
                raise ValueError(f"Element block {block!r} has already been assigned properties")
        for b in self.mesh._blocks:
            if b.name == block:
                blk = ElementBlock.from_topo_block(b, element, material)
                self._blocks.append(blk)
                break
        else:
            raise ValueError(f"Element block {block!r} is not defined")
