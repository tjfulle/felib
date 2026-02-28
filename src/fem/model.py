import logging
from dataclasses import dataclass
from dataclasses import field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .block import ElementBlock
from .collections import Map
from .element import Element
from .material import Material
from .mesh import Mesh
from .step import Step

# Combined type union for array-like input
ArrayLike = NDArray | Sequence


logger = logging.getLogger(__name__)


@dataclass
class Model:
    """
    Finite element model container class.

    Holds mesh data, block definitions, degrees of freedom mappings,
    loading steps, solution vectors, and methods for assembly and solving.

    Args:
        name: Human-readable model name.
        mesh: Mesh object containing node coordinates and connectivity.
        blocks: List of ElementBlock objects (one per element block).
        num_dof: Total number of degrees of freedom in the model.
        dof_map: Global dof mapping array.
        block_dof_map: Array mapping block indices to dof indices.
    """

    name: str
    mesh: Mesh
    blocks: list[ElementBlock]
    num_dof: int
    dof_map: NDArray
    node_freedom_table: NDArray
    node_freedom_types: list[int]
    block_dof_map: NDArray

    # Solution and residual storage
    u: NDArray = field(init=False)
    R: NDArray = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialize internal solution and residual arrays.

        Creates two time levels for displacements (self.u) and residuals (self.R)
        with shape (2, num_dof). The first index holds the previous state,
        and the second index holds the current state.
        """
        self.u = np.zeros((2, self.num_dof), dtype=float)
        self.R = np.zeros((2, self.num_dof), dtype=float)

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
