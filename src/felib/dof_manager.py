import re
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .model import Model


class DOF(IntEnum):
    ux, uy, uz = 0, 1, 2
    rx, ry, rz = 3, 4, 5
    T = 6

    def __init__(self, value):
        super().__init__()
        rx = "^[ur][xyz]$"
        self.label = self.name if not re.search(rx, self.name) else f"{self.name[0]}.{self.name[1]}"


class DOFManager:
    def __init__(self, model: "Model") -> None:
        self.ndim = model.ndim
        self._node_freedom_table: NDArray = np.empty((0, 0), dtype=int)
        self._node_dof_types: list[DOF] = []
        self._node_dof_cols: dict[DOF, int] = {}
        self._block_dof_maps: dict[int, NDArray] = {}
        self._dof_map: NDArray = np.empty((0, 0), dtype=int)
        self._dof_types: NDArray = np.empty(0, dtype=int)
        self._ndof: int = -1
        self.build(model)

    def global_dof(self, node: int, ldof: int) -> int:
        return self._dof_map[node, ldof]

    def dof_index(self, dof_type: DOF) -> int:
        return int(self.node_dof_cols[dof_type])

    def node_freedom_type(self, local_dof: int) -> DOF:
        return self.node_dof_types[local_dof]

    @property
    def size(self) -> int:
        assert self._ndof > 0
        return self.ndof

    @property
    def nnode(self) -> int:
        return self.dof_map.shape[0]

    @property
    def nlocal_dofs(self) -> int:
        return self.dof_map.shape[1]

    def shape(self) -> tuple[int, int]:
        return (self.nnode, self.nlocal_dofs)

    def types(self) -> NDArray:
        return self.dof_types

    def block_freedom_table(self, blockno: int) -> NDArray:
        table = self.block_dof_maps[blockno]
        return table

    @property
    def node_freedom_table(self) -> NDArray:
        return self._node_freedom_table

    @property
    def node_dof_types(self) -> list[DOF]:
        return self._node_dof_types

    @property
    def node_dof_cols(self) -> dict[DOF, int]:
        return self._node_dof_cols

    @property
    def block_dof_maps(self) -> dict[int, NDArray]:
        return self._block_dof_maps

    @property
    def dof_map(self) -> NDArray:
        """Maps (node, local DOF index in node_freedom_table) -> global DOF index"""
        return self._dof_map

    @property
    def dof_types(self) -> NDArray:
        return self._dof_types

    @property
    def ndof(self) -> int:
        return self._ndof

    def build(self, model: "Model") -> None:
        """
        Build all DOF-related tables for the model.

        Steps:
            1) Build node_freedom_table and node_dof_types
            2) Build global DOF map: dof_map[node, local_dof]
            3) Build per-block DOF map: block_dof_map[blockno][local_dof]
        """
        # 1) Node-level DOFs
        self.build_node_freedom_table(model)

        # 2) Global DOF numbering
        self.build_dof_map()

        # 3) Precompute block → global DOFs
        self.build_block_dof_maps(model)

    def build_node_freedom_table(self, model: "Model") -> None:
        """
        Build the model-level node freedom table.

        Produces:
            self._node_freedom_table : ndarray[int] of shape (nnode, n_active_dofs)
                1 if the DOF is active at the node, 0 otherwise
            self._node_dof_types : ndarray[int] of length n_active_dofs
                Physical DOF type corresponding to each column (Ux, Uy, ..., T)
            self._node_dof_cols : dict[int, int] of length n_active_dofs
                Physical DOF type corresponding to each column (Ux, Uy, ..., T)
            self.ndof : int
                Total number of active DOFs in the model
        """

        # -----------------------------
        # 1) Determine max DOF index used across all blocks
        # -----------------------------
        max_dof_idx = -1
        for block in model._blocks:
            for node_dofs in block.element.node_freedom_table:
                max_dof_idx = max(max_dof_idx, max(node_dofs))
        ncol = max_dof_idx + 1  # number of physical DOF types used

        nnode = model.mesh.coords.shape[0]

        # -----------------------------
        # 2) Build full node signature (nnode x ncol)
        # -----------------------------
        node_sig_full = np.zeros((nnode, ncol), dtype=int)

        for block in model._blocks:
            for elem_nodes in block.connect:
                for i, block_node in enumerate(elem_nodes):
                    gid = block.node_map[block_node]
                    lid = model.mesh.node_map.local(gid)
                    for col in block.element.node_freedom_table[i]:
                        node_sig_full[lid, col] = 1

        # -----------------------------
        # 3) Identify active columns
        # -----------------------------
        active_cols = np.where(node_sig_full.sum(axis=0) > 0)[0]

        # -----------------------------
        # 4) Compress node signature and store
        # -----------------------------
        self._node_freedom_table = node_sig_full[:, active_cols].copy()
        self._node_dof_types = [DOF(_) for _ in sorted(active_cols)]
        self._node_dof_cols = {t: i for i, t in enumerate(self._node_dof_types)}
        self._ndof = int(self._node_freedom_table.sum())

    def build_dof_map(self) -> None:
        """
        Build global DOF numbering for the model.

        Produces:
            self.dof_map: ndarray[int] of shape (nnode, n_active_dofs)
                Maps (node, local DOF index in node_freedom_table) -> global DOF index
        """
        nnode, n_active = self._node_freedom_table.shape
        self._dof_map = -np.ones((nnode, n_active), dtype=int)

        # Flatten node_freedom_table
        mask = self._node_freedom_table.ravel()
        global_dofs = np.arange(mask.sum(), dtype=int)

        # Assign global DOF indices where DOFs are active
        table = self._node_freedom_table
        self._dof_map[table == 1] = global_dofs

        # DOFs associated with displacment
        dof_types = np.zeros(global_dofs.size, dtype=int)
        node_types_expanded = np.tile(self._node_dof_types, (nnode, 1))
        dof_types[:] = node_types_expanded.ravel()[mask]
        self._dof_types = dof_types

    def build_block_dof_maps(self, model: "Model") -> None:
        """
        Build a precomputed block DOF map:
            block_dof_map[blockno][local_dof] = global DOF label

        After this, `block_freedom_table(blockno)` is simply:
            bft = self.block_dof_map[blockno, :n_block_dof]
        """
        for blockno, block in enumerate(model._blocks):
            dof_per_node = block.element.dof_per_node
            nnode = block.nnode
            n_block_dof = nnode * dof_per_node
            block_dof_map = -np.ones(n_block_dof, dtype=int)
            local_dof_idx = 0
            for i in range(nnode):
                gid = block.node_map[i]
                lid = model.mesh.node_map.local(gid)
                for dof_type in block.element.node_freedom_table[0]:
                    col = self._node_dof_cols[DOF(dof_type)]
                    gdof = self._dof_map[lid, col]
                    block_dof_map[local_dof_idx] = gdof
                    local_dof_idx += 1
            if np.any(block_dof_map < 0):
                raise RuntimeError(f"Unassigned node freedom in {block.name}")
            self._block_dof_maps[blockno] = block_dof_map
