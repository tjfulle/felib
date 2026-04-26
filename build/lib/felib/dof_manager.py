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

    # TODO: Homogeneous MPC support
    # The current DOFManager builds a full/global DOF numbering for all active
    # DOFs in the model. To support homogeneous-MPC tied contact we will need
    # a mechanism to represent a reduced DOF space (mains) and to map/expand
    # secondary DOFs using a transformation matrix `T` such that
    #
    #   u_full = T @ u_reduced
    #
    # Suggested extension points (add methods here later):
    # - `build_mpc_map(mpc_constraints)` : compute reduced DOF indices and
    #    bookkeeping for main DOFs. Should not change existing `dof_map`
    #    but provide a `reduced_dof_map` view and mapping helpers.
    # - `expand_solution(u_reduced, T)` : produce the full-length displacement
    #    vector from a reduced solution (callable by the Step after solve).
    # - `reduced_size()` : return number of DOFs in reduced system.
    #
    # Important considerations:
    # - DOF ordering (ux, uy, ...) must be consistent when building `T`.
    # - Keep `dof_map` (full indexing) intact so existing code that expects
    #   full DOF indices continues to work; provide the reduced mapping as
    #   an auxiliary structure to the solver/assembly routines.

    def build_mpc_map(self, mpc_constraints) -> None:
        """
        Placeholder for future implementation of MPC mapping.
        Method would compute reduced DOF indices and bookkeeping for 
        main DOFs based on provided MPC constraints. Should not change existing
        DOF map, but provide a 'reduced_dof_map' view and mapping helpers
        """

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

    def build_mpc_map(self, mpcs: list) -> tuple[NDArray, NDArray]:
        """
        Build an MPC transformation matrix `T` and offset vector for a list
        of homogeneous multi-point constraints (MPCs).

        Expected `mpcs` format (simple, unambiguous): a list of tuples
            (secondary_dof, mains, offset)
        where
            - `secondary_dof` is an int global DOF index (0..ndof-1)
            - `mains` is a list of `(main_dof:int, coeff:float)` tuples
            - `offset` is a float (often 0.0 for homogeneous ties)

        The produced transform `T` has shape (ndof, n_reduced) and satisfies

            u_full = T @ u_reduced + offset_vector

        where `u_reduced` contains values for all DOFs that are *not* secondaries
        (i.e. independent DOFs). The reduced DOFs ordering is the ascending
        list of full DOF indices that are not secondaries.

        Returns
        -------
        T : NDArray
            Transformation matrix, shape (ndof, n_reduced)
        offset : NDArray
            Offset vector of length `ndof` (usually zeros for homogeneous MPC)

        Notes
        -----
        - This implementation assumes mains are not themselves declared as
          secondaries. Handling chains of dependent MPCs requires graph ordering
          or elimination and is not implemented here.
        """
        ndof = int(self._ndof)
        if ndof <= 0:
            raise RuntimeError("DOFManager: build node freedom table before building MPC map")

        # Mark secondary DOFs and collect mains/offsets
        is_secondary = np.zeros(ndof, dtype=bool)
        offset = np.zeros(ndof, dtype=float)
        main_set = set()

        for entry in mpcs:
            if len(entry) == 3:
                secondary, mains, off = entry
            elif len(entry) == 2:
                secondary, mains = entry
                off = 0.0
            else:
                raise ValueError("MPC entry must be (secondary, mains[, offset])")
            secondary = int(secondary)
            is_secondary[secondary] = True
            offset[secondary] = float(off)
            for m, c in mains:
                main_set.add(int(m))

        # Basic sanity: mains must not be secondaries in this simple implementation
        mains_list = sorted(main_set)
        if any(is_secondary[np.array(mains_list, dtype=int)]):
            raise RuntimeError("MPC mains that are also secondaries are not supported by this implementation")

        # Reduced DOFs are all DOFs that are not declared as secondaries
        reduced_dofs = [i for i in range(ndof) if not is_secondary[i]]
        n_reduced = len(reduced_dofs)
        if n_reduced == 0:
            raise RuntimeError("No reduced DOFs found (all DOFs are secondaries)")

        # Build transform matrix T (ndof x n_reduced)
        T = np.zeros((ndof, n_reduced), dtype=float)
        reduced_index = {d: i for i, d in enumerate(reduced_dofs)}

        # Identity for independent DOFs
        for i, d in enumerate(reduced_dofs):
            T[d, i] = 1.0

        # Fill secondary rows using provided main coefficients
        for entry in mpcs:
            if len(entry) == 3:
                secondary, mains, _ = entry
            else:
                secondary, mains = entry
            secondary = int(secondary)
            for m, coeff in mains:
                m = int(m)
                if m not in reduced_index:
                    raise RuntimeError(f"Master DOF {m} not available in reduced DOF set")
                T[secondary, reduced_index[m]] = float(coeff)

        # Store bookkeeping
        self._mpc_T = T
        self._mpc_offset = offset
        self._mpc_reduced_dofs = reduced_dofs

        return T, offset

    def compute_transform(self, fdofs: NDArray | None = None) -> tuple[NDArray, NDArray]:
        """Return transform `T_f` and offset restricted to free DOFs `fdofs`.

        If `fdofs` is None the full transform `T` and full offset are returned.
        """
        if not hasattr(self, "_mpc_T"):
            raise RuntimeError("MPC transform has not been built. Call build_mpc_map first.")
        T = self._mpc_T
        offset = self._mpc_offset
        if fdofs is None:
            return T, offset
        fdofs = np.asarray(fdofs, dtype=int)
        return T[fdofs, :], offset[fdofs]

    def expand_solution(self, u_reduced: NDArray) -> NDArray:
        """Expand reduced solution into full DOF vector using stored transform.

        Returns `u_full = T @ u_reduced + offset`.
        """
        if not hasattr(self, "_mpc_T"):
            raise RuntimeError("MPC transform has not been built. Call build_mpc_map first.")
        return self._mpc_T.dot(u_reduced) + self._mpc_offset

    def reduced_size(self) -> int:
        if not hasattr(self, "_mpc_reduced_dofs"):
            raise RuntimeError("MPC transform has not been built. Call build_mpc_map first.")
        return len(self._mpc_reduced_dofs)
