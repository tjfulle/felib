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
        self._mpc_T: NDArray | None = None
        self._mpc_offset: NDArray | None = None
        self._mpc_reduced_dofs: NDArray = np.empty(0, dtype=int)
        self._mpc_slave_dofs: NDArray = np.empty(0, dtype=int)
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

    @property
    def has_mpc_transform(self) -> bool:
        return self._mpc_T is not None

    def clear_mpc_map(self) -> None:
        self._mpc_T = None
        self._mpc_offset = None
        self._mpc_reduced_dofs = np.empty(0, dtype=int)
        self._mpc_slave_dofs = np.empty(0, dtype=int)

    def build_mpc_map(self, mpcs: list, *, tol: float = 1e-14) -> tuple[NDArray, NDArray]:
        """
        Build and store an MPC transformation matrix `T` and offset vector.

        Accepted MPC formats are either normalized tuples
        ``(slave_dof, masters, offset)`` or compiled equation rows
        ``[dof, coeff, dof, coeff, ..., rhs]``. Equation rows are normalized
        by treating the first nonzero term as the slave DOF.

        The produced transform `T` has shape (ndof, n_reduced) and satisfies

            u_full = T @ u_reduced + offset_vector

        where `u_reduced` contains values for all DOFs that are *not* slaves
        (i.e. independent DOFs). The reduced DOFs ordering is the ascending
        list of full DOF indices that are not slaves.

        Returns
        -------
        T : NDArray
            Transformation matrix, shape (ndof, n_reduced)
        offset : NDArray
            Offset vector of length `ndof` (usually zeros for homogeneous MPC)

        Notes
        -----
        - This implementation assumes masters are not themselves declared as
          slaves. Handling chains of dependent MPCs requires graph ordering
          or elimination and is not implemented here.
        """
        self.clear_mpc_map()
        ndof = int(self._ndof)
        if ndof <= 0:
            raise RuntimeError("DOFManager: build node freedom table before building MPC map")
        if not mpcs:
            return np.eye(ndof, dtype=float), np.zeros(ndof, dtype=float)

        normalized = [self._normalize_mpc_entry(entry, tol=tol) for entry in mpcs]
        is_slave = np.zeros(ndof, dtype=bool)
        offset = np.zeros(ndof, dtype=float)
        master_set: set[int] = set()

        for slave, masters, off in normalized:
            slave = int(slave)
            self._validate_dof(slave)
            if is_slave[slave]:
                raise RuntimeError(f"Slave DOF {slave} is constrained by multiple MPCs")
            is_slave[slave] = True
            offset[slave] = float(off)
            for m, c in masters:
                master = int(m)
                self._validate_dof(master)
                master_set.add(master)

        # Basic sanity: masters must not be slaves in this simple implementation
        masters_list = sorted(master_set)
        if any(is_slave[np.array(masters_list, dtype=int)]):
            raise RuntimeError(
                "MPC masters that are also slaves are not supported by this implementation"
            )

        # Reduced DOFs are all DOFs that are not declared as slaves
        reduced_dofs = [i for i in range(ndof) if not is_slave[i]]
        n_reduced = len(reduced_dofs)
        if n_reduced == 0:
            raise RuntimeError("No reduced DOFs found (all DOFs are slaves)")

        # Build transform matrix T (ndof x n_reduced)
        T = np.zeros((ndof, n_reduced), dtype=float)
        reduced_index = {d: i for i, d in enumerate(reduced_dofs)}

        # Identity for independent DOFs
        for i, d in enumerate(reduced_dofs):
            T[d, i] = 1.0

        # Fill slave rows using provided master coefficients
        for slave, masters, _ in normalized:
            slave = int(slave)
            for m, coeff in masters:
                m = int(m)
                if m not in reduced_index:
                    raise RuntimeError(f"Master DOF {m} not available in reduced DOF set")
                T[slave, reduced_index[m]] = float(coeff)

        # Store bookkeeping
        self._mpc_T = T
        self._mpc_offset = offset
        self._mpc_reduced_dofs = np.array(reduced_dofs, dtype=int)
        self._mpc_slave_dofs = np.flatnonzero(is_slave)

        return T, offset

    def _normalize_mpc_entry(
        self, entry, *, tol: float
    ) -> tuple[int, list[tuple[int, float]], float]:
        if self._is_normalized_mpc(entry):
            if len(entry) == 3:
                slave, masters, offset = entry
            else:
                slave, masters = entry
                offset = 0.0
            return int(slave), [(int(m), float(c)) for m, c in masters], float(offset)

        if len(entry) < 5 or len(entry) % 2 == 0:
            raise ValueError("Compiled MPC equation must be [dof, coeff, ..., rhs]")

        rhs = float(entry[-1])
        order: list[int] = []
        terms: dict[int, float] = {}
        for i in range(0, len(entry) - 1, 2):
            dof = int(entry[i])
            coeff = float(entry[i + 1])
            if dof not in terms:
                order.append(dof)
            terms[dof] = terms.get(dof, 0.0) + coeff

        active_terms = [(dof, terms[dof]) for dof in order if abs(terms[dof]) > tol]
        if len(active_terms) < 2:
            raise ValueError("MPC equation must have at least two nonzero terms")

        slave, slave_coeff = active_terms[0]
        masters = [(master, -coeff / slave_coeff) for master, coeff in active_terms[1:]]
        offset = rhs / slave_coeff
        return slave, masters, offset

    @staticmethod
    def _is_normalized_mpc(entry) -> bool:
        if not isinstance(entry, (list, tuple)):
            return False
        if len(entry) not in (2, 3):
            return False
        masters = entry[1]
        if not isinstance(masters, (list, tuple)):
            return False
        return all(isinstance(master, (list, tuple)) and len(master) == 2 for master in masters)

    def _validate_dof(self, dof: int) -> None:
        if dof < 0 or dof >= self._ndof:
            raise ValueError(f"DOF {dof} is outside the active range 0..{self._ndof - 1}")

    def can_apply_mpc_reduction(self, ddofs: NDArray | list[int] | None = None) -> bool:
        if not self.has_mpc_transform:
            return False
        if ddofs is None:
            return True
        ddofs = np.asarray(ddofs, dtype=int)
        return not np.any(np.isin(ddofs, self._mpc_slave_dofs))

    def step_transform(
        self,
        ddofs: NDArray | list[int],
        dvals: NDArray | list[float],
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Return ``(T_step, offset_step, reduced_free_dofs)`` for a step.

        ``T_step`` maps only unconstrained independent unknowns to the full DOF
        vector. Prescribed independent DOFs are propagated into ``offset_step``.
        """
        if self._mpc_T is None or self._mpc_offset is None:
            raise RuntimeError("MPC transform has not been built. Call build_mpc_map first.")
        ddofs = np.asarray(ddofs, dtype=int)
        dvals = np.asarray(dvals, dtype=float)
        if len(ddofs) != len(dvals):
            raise ValueError("ddofs and dvals must have the same length")
        if np.any(np.isin(ddofs, self._mpc_slave_dofs)):
            raise RuntimeError("Prescribed slave DOFs cannot be eliminated with MPC reduction")

        reduced_dofs = self._mpc_reduced_dofs
        prescribed_reduced = np.isin(reduced_dofs, ddofs)
        free_reduced = ~prescribed_reduced
        free_cols = np.flatnonzero(free_reduced)
        prescribed_cols = np.flatnonzero(prescribed_reduced)

        offset = self._mpc_offset.copy()
        if len(prescribed_cols):
            dval_by_dof = {int(dof): float(value) for dof, value in zip(ddofs, dvals)}
            q_known = np.array(
                [dval_by_dof[int(reduced_dofs[col])] for col in prescribed_cols],
                dtype=float,
            )
            offset = offset + self._mpc_T[:, prescribed_cols] @ q_known

        return self._mpc_T[:, free_cols], offset, reduced_dofs[free_cols]

    def reduced_initial_values(
        self,
        u_full: NDArray,
        ddofs: NDArray | list[int],
    ) -> NDArray:
        _, _, reduced_free_dofs = self.step_transform(ddofs, np.zeros(len(ddofs), dtype=float))
        return np.asarray(u_full, dtype=float)[reduced_free_dofs]

    def expand_step_solution(
        self,
        u_reduced: NDArray,
        ddofs: NDArray | list[int],
        dvals: NDArray | list[float],
    ) -> NDArray:
        T, offset, _ = self.step_transform(ddofs, dvals)
        u_reduced = np.asarray(u_reduced, dtype=float)
        if T.shape[1] != len(u_reduced):
            raise ValueError(f"Reduced solution has length {len(u_reduced)}, expected {T.shape[1]}")
        return T @ u_reduced + offset

    def compute_transform(self, fdofs: NDArray | None = None) -> tuple[NDArray, NDArray]:
        """Return transform `T_f` and offset restricted to free DOFs `fdofs`.

        If `fdofs` is None the full transform `T` and full offset are returned.
        """
        if self._mpc_T is None or self._mpc_offset is None:
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
        if self._mpc_T is None or self._mpc_offset is None:
            raise RuntimeError("MPC transform has not been built. Call build_mpc_map first.")
        return self._mpc_T.dot(u_reduced) + self._mpc_offset

    def reduced_size(self) -> int:
        if not self.has_mpc_transform:
            raise RuntimeError("MPC transform has not been built. Call build_mpc_map first.")
        return len(self._mpc_reduced_dofs)
