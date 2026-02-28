from dataclasses import dataclass
from dataclasses import field
from typing import Sequence

import exodusii
import numpy as np
from numpy.typing import NDArray

from .block import ElementBlock
from .collections import Map
from .mesh import Mesh
from .step import Step

# Combined type union for array-like input
ArrayLike = NDArray | Sequence


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
        steps: Optional list of Step objects representing analysis steps.
    """

    name: str
    mesh: Mesh
    blocks: list[ElementBlock]
    num_dof: int
    dof_map: NDArray
    node_freedom_table: NDArray
    node_freedom_types: NDArray
    block_dof_map: NDArray
    steps: list[Step] = field(default_factory=list)

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

    def add_step(self, step: Step) -> None:
        """
        Append a new analysis step.

        Args:
            step: A Step object representing boundary conditions, loads, and solver settings.
        """
        self.steps.append(step)

    def solve(self) -> None:
        """
        Run through all analysis steps and solve.

        For each step, triggers Step.solve(), advances state, and writes results to
        the Exodus output file.
        """
        file = ExodusFile(self)
        for i, step in enumerate(self.steps):
            step.solve(self)
            self.advance_state()
            for block in self.blocks:
                block.advance_state()
            file.update(i + 1, step)

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


class ExodusFile:
    """
    Wrapper for Exodus II file writing.

    Handles initialization, element block definitions, field variable
    definitions, and writing of results for each time step.
    """

    def __init__(self, model: Model) -> None:
        """
        Create and initialize the Exodus file.

        Args:
            model: Model object to write output for.
        """
        fname = "_".join(model.name.split()) + ".exo"
        file = exodusii.exo_file(fname, mode="w")
        file.put_init(
            f"fem solution for {model.name}",
            num_dim=model.coords.shape[1],
            num_nodes=model.coords.shape[0],
            num_elem=model.connect.shape[0],
            num_elem_blk=len(model.blocks),
            num_node_sets=len(model.mesh.nodesets),
            num_side_sets=len(model.mesh.sidesets),
        )

        # Write coordinate and connectivity data
        coord_names = [f"disp{'xyz'[i]}" for i in range(model.coords.shape[1])]
        file.put_coord_names(coord_names)
        file.put_coords(model.coords)
        file.put_map(model.elem_map.lid_to_gid)
        file.put_node_id_map(model.node_map.lid_to_gid)

        for i, block in enumerate(model.blocks):
            file.put_element_block(
                i + 1,
                block.element.family,
                num_block_elems=block.connect.shape[0],
                num_nodes_per_elem=block.element.nnode,
                num_faces_per_elem=1,
                num_edges_per_elem=3,
            )
            file.put_element_conn(i + 1, block.connect + 1)
            file.put_element_block_name(i + 1, f"Block-{i + 1}")

        # Build unique list of element variable names and truth table
        elem_vars: list[str] = []
        for block in model.blocks:
            for elem_var in block.element_variable_names():
                if elem_var not in elem_vars:
                    elem_vars.append(elem_var)
        truth_tab = np.zeros((len(model.blocks), len(elem_vars)), dtype=int)
        for i, block in enumerate(model.blocks):
            for j, elem_var in enumerate(elem_vars):
                if elem_var in block.element_variable_names():
                    truth_tab[i, j] = 1
        file.put_element_variable_params(len(elem_vars))
        file.put_element_variable_names(elem_vars)
        file.put_element_variable_truth_table(truth_tab)

        # Write node sets
        nodeset_id = 1
        for name, lids in model.nodesets.items():
            gids = [model.node_map[lid] for lid in lids]
            file.put_node_set_param(nodeset_id, len(gids), 0)
            file.put_node_set_name(nodeset_id, str32(name))
            file.put_node_set_nodes(nodeset_id, gids)
            nodeset_id += 1

        # Write side sets
        sideset_id = 1
        #        for name, ss in model.sidesets.items():
        #            file.put_side_set_param(sideset_id, len(ss), 0)
        #            file.put_side_set_name(sideset_id, str32(name))
        #            gids = [model.elem_map[_[0]] for _ in ss]
        #            sides = [_[1] + 1 for _ in ss]
        #            file.put_side_set_sides(sideset_id, gids, sides)
        #            sideset_id += 1

        # Setup result variables
        file.put_global_variable_params(1)
        file.put_global_variable_names(["time_step"])
        file.put_time(1, 0.0)
        file.put_global_variable_values(1, np.zeros(1, dtype=float))

        file.put_node_variable_params(model.coords.shape[1])
        displ_names = [self.displ_name(i) for i in range(model.coords.shape[1])]
        file.put_node_variable_names(displ_names)

        for i in range(model.coords.shape[1]):
            file.put_node_variable_values(1, self.displ_name(i), model.u[0, i::2])

        # Write initial element variable values
        for j, block in enumerate(model.blocks):
            for name, value in block.element_variable_values():
                file.put_element_variable_values(1, j + 1, name, value)

        self.file = file
        self.model = model

    def update(self, step_no: int, step: Step) -> None:
        """
        Write updated values for a new time step.

        Args:
            step_no: Index of current step.
            step: Step object containing updated results.
        """
        file = self.file
        model = self.model

        file.put_time(step_no + 1, step.start + step.period)

        for i in range(model.coords.shape[1]):
            file.put_node_variable_values(step_no + 1, self.displ_name(i), model.u[0, i::2])

        for j, block in enumerate(model.blocks):
            for name, value in block.element_variable_values():
                file.put_element_variable_values(step_no + 1, j + 1, name, value)

    def displ_name(self, i: int) -> str:
        """Generate node variable name for displacement dimension i."""
        return f"displ{'xyz'[i]}"


def str32(string: str) -> str:
    """
    Format string to 32 characters for Exodus set names.

    Pads or truncates to ensure 32-character width.
    """
    return f"{string:32}"
