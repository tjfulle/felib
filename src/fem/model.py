import logging
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .block import ElementBlock
from .collections import Map
from .element import Element
from .material import Material
from .mesh import Mesh
from .pytools import _require_unfrozen
from .pytools import frozen_property
from .step import Step

# Combined type union for array-like input
ArrayLike = NDArray | Sequence


logger = logging.getLogger(__name__)


class Model:
    """Finite element model container class."""

    def __init__(self, mesh: "Mesh", name: str = "Model-1") -> None:
        self.name = name
        self.mesh = mesh
        self.mesh.freeze()

        self._builder = _ModelBuilder(self)
        self._frozen = False

        # Populated by builder
        self._blocks: list[ElementBlock] = []
        self._node_freedom_table: NDArray = np.empty((0, 0), dtype=int)
        self._node_freedom_types: list[int] = []
        self._block_dof_map: NDArray = np.empty((0, 0), dtype=int)
        self._dof_map: NDArray = np.empty((0, 0), dtype=int)
        self._num_dof: int = -1

        self.u = np.empty((0, 0), dtype=float)
        self.R = np.empty((0, 0), dtype=float)

    def freeze(self) -> None:
        if not self._frozen:
            self._builder.build()
            self._frozen = True
            self.u = np.zeros((2, self.num_dof), dtype=float)
            self.R = np.zeros((2, self.num_dof), dtype=float)

    @frozen_property
    def blocks(self) -> list[ElementBlock]:
        return self._blocks

    @frozen_property
    def node_freedom_table(self) -> NDArray:
        return self._node_freedom_table

    @frozen_property
    def node_freedom_types(self) -> list[int]:
        return self._node_freedom_types

    @frozen_property
    def block_dof_map(self) -> NDArray:
        return self._block_dof_map

    @frozen_property
    def dof_map(self) -> NDArray:
        return self._dof_map

    @frozen_property
    def num_dof(self) -> int:
        return self._num_dof

    @property
    def nnode(self) -> int:
        """Return the number of global nodes in the mesh."""
        return self.mesh.coords.shape[0]

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
        self._builder.assign_properties(block=block, element=element, material=material)

    def advance_state(self) -> None:
        """
        Advance the state from current solution to previous solution.

        Copies the contents of self.u[1] -> self.u[0]
        and self.R[1] -> self.R[0], preparing for next step.
        """
        self.u[0, :] = self.u[1, :]
        self.R[0, :] = self.R[1, :]
        for block in self.blocks:
            block.advance_state()

    def assemble(
        self,
        step: Step,
        increment: int,
        time: Sequence[float],
        dt: float,
        u: NDArray,
        du: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """
        Global matrix and residual assembly.

        Calls ElementBlock.assemble() for each block, inserting into global stiffness
        matrix and force vector.

        Args:
            step: Current analysis step.
            increment: Sub-increment index.
            time: Current time history.
            dt: Time step size.
            u: Current displacement vector.
            du: Displacement increment vector.

        Returns:
            Tuple of (K_global, R_global)
        """
        K = np.zeros((self.num_dof, self.num_dof), dtype=float)
        R = np.zeros(self.num_dof, dtype=float)
        for b, block in enumerate(self.blocks):
            bft = self.block_freedom_table(b)
            kb, rb = block.assemble(
                step.number,
                increment,
                time,
                dt,
                u[bft],
                du[bft],
                dloads=step.dloads.get(b),
                dsloads=step.dsloads.get(b),
                rloads=step.rloads.get(b),
            )
            K[np.ix_(bft, bft)] += kb
            R[bft] += rb
        return K, R

    def block_freedom_table(self, blockno: int) -> NDArray:
        """
        Return the compact global DOFs for the entire block.
        """
        dof_per_node = self.blocks[blockno].element.dof_per_node
        nnode = self.blocks[blockno].num_nodes
        n_block_dof = nnode * dof_per_node
        return self.block_dof_map[blockno, :n_block_dof]


class _ModelBuilder:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.assembled = False

    def assign_properties(self, *, block: str, element: Element, material: Material) -> None:
        for blk in self.model._blocks:
            if blk.name == block:
                raise ValueError(f"Element block {block!r} has already been assigned properties")
        for b in self.model.mesh._blocks:
            if b.name == block:
                blk = ElementBlock.from_topo_block(b, element, material)
                self.model._blocks.append(blk)
                break
        else:
            raise ValueError(f"Element block {block!r} is not defined")

    def build(self) -> None:
        if self.assembled:
            raise ValueError("ModelBuilder.build() already called")
        blocks = {block.name for block in self.model.mesh._blocks}
        if missing := blocks.difference({block.name for block in self.model._blocks}):
            raise ValueError(f"The following blocks have not been assigned properties: {missing}")
        self.build_dof_maps()
        self.assembled = True
        return

    def build_node_freedom_table(self) -> None:
        """
        Build the model-level node freedom table.

        Produces:
            self._node_freedom_table : ndarray[int] of shape (nnode, n_active_dofs)
                1 if the DOF is active at the node, 0 otherwise
            self._node_freedom_types : ndarray[int] of length n_active_dofs
                Physical DOF type corresponding to each column (Ux, Uy, ..., T)
            self.num_dof : int
                Total number of active DOFs in the model
        """

        # -----------------------------
        # 1) Determine max DOF index used across all blocks
        # -----------------------------
        max_dof_idx = -1
        for block in self.model._blocks:
            for node_dofs in block.element.node_freedom_table:
                max_dof_idx = max(max_dof_idx, max(node_dofs))
        ncol = max_dof_idx + 1  # number of physical DOF types used

        nnode = self.model.mesh.coords.shape[0]

        # -----------------------------
        # 2) Build full node signature (nnode x ncol)
        # -----------------------------
        node_sig_full = np.zeros((nnode, ncol), dtype=int)

        for block in self.model._blocks:
            for elem_nodes in block.connect:
                for i, block_node in enumerate(elem_nodes):
                    gid = block.node_map[block_node]
                    lid = self.model.mesh.node_map.local(gid)
                    for col in block.element.node_freedom_table[i]:
                        node_sig_full[lid, col] = 1

        # -----------------------------
        # 3) Identify active columns
        # -----------------------------
        active_cols = np.where(node_sig_full.sum(axis=0) > 0)[0]

        # -----------------------------
        # 4) Compress node signature and store
        # -----------------------------
        self.model._node_freedom_table = node_sig_full[:, active_cols].copy()
        self.model._node_freedom_types = active_cols.tolist()
        self.model._num_dof = int(self.model._node_freedom_table.sum())

    def build_dof_map(self) -> None:
        """
        Build global DOF numbering for the model.

        Produces:
            self.dof_map: ndarray[int] of shape (nnode, n_active_dofs)
                Maps (node, local DOF index in node_freedom_table) -> global DOF index
        """
        nnode, n_active = self.model._node_freedom_table.shape
        self.model._dof_map = -np.ones((nnode, n_active), dtype=int)

        # Flatten node_freedom_table
        flat_mask = self.model._node_freedom_table.ravel()
        global_dofs = np.arange(flat_mask.sum(), dtype=int)

        # Assign global DOF indices where DOFs are active
        self.model._dof_map[self.model._node_freedom_table == 1] = global_dofs

    def build_block_dof_map(self) -> None:
        """
        Build a precomputed block DOF map:
            block_dof_map[blockno, local_dof] = global DOF label

        After this, `block_freedom_table(blockno)` is simply:
            bft = self.block_dof_map[blockno, :n_block_dof]
        """
        # Determine max DOFs per block
        max_block_dofs = max(b.num_dof for b in self.model._blocks)

        # Initialize table with -1
        self.model._block_dof_map = -np.ones((len(self.model._blocks), max_block_dofs), dtype=int)

        for blockno, block in enumerate(self.model._blocks):
            local_dof_idx = 0
            for i in range(block.num_nodes):
                gid = block.node_map[i]
                lid = self.model.mesh.node_map.local(gid)
                for dof_type in block.element.node_freedom_table[0]:
                    col = self.model._node_freedom_types.index(dof_type)
                    gdof = self.model._dof_map[lid, col]
                    self.model._block_dof_map[blockno, local_dof_idx] = gdof
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
