from typing import TYPE_CHECKING
from typing import Any

import exodusii
import numpy as np
from numpy.typing import NDArray

from . import constants
from .block import ElementBlock
from .collections import ElementBlockData
from .step import CompiledStep
from .step import DirectStep
from .step import HeatTransferStep
from .step import StaticStep
from .step import Step
from .step.staggered import StaggeredStep

if TYPE_CHECKING:
    from .model import Model


class Simulation:
    def __init__(self, model: "Model") -> None:
        self.model: "Model" = model
        self.model.freeze()
        self.steps: list[Step] = []
        self.csteps: list[CompiledStep] = []

        # Solution and residual storage
        self.dofs: NDArray = np.zeros((2, self.model.ndof))
        self.flux: NDArray = np.zeros((2, self.model.ndof))
        self.ebdata: list[ElementBlockData] = []
        b: ElementBlock
        for b in model.blocks:
            d = ElementBlockData(b.name, b.nelem, b.element.npts, b.element_variable_names())
            self.ebdata.append(d)

    def advance_state(self) -> None:
        """
        Advance the state from current solution to previous solution.

        Copies the contents of self.u[1] -> self.u[0]
        and self.R[1] -> self.R[0], preparing for next step.
        """
        self.dofs[0, :] = self.dofs[1, :]
        self.flux[0, :] = self.flux[1, :]
        for d in self.ebdata:
            d.advance_state()

    def static_step(
        self, name: str | None = None, period: float = 1.0, **options: Any
    ) -> StaticStep:
        name = name or f"step-{len(self.steps)}"
        step = StaticStep(name=name, period=period, **options)
        self.steps.append(step)
        return step

    def direct_step(self, name: str | None = None, period: float = 1.0) -> DirectStep:
        name = name or f"step-{len(self.steps)}"
        step = DirectStep(name=name, period=period)
        self.steps.append(step)
        return step

    def heat_transfer_step(self, name: str | None = None, period: float = 1.0) -> HeatTransferStep:
        name = name or f"step-{len(self.steps)}"
        step = HeatTransferStep(name=name, period=period)
        self.steps.append(step)
        return step
    def staggered_step(
    self, name: str | None = None, period: float = 1.0, **options: Any
) -> StaggeredStep:
        name = name or f"step-{len(self.steps)}"
        step = StaggeredStep(name=name, period=period, **options)
        self.steps.append(step)
        return step

    def run(self) -> None:
        """
        Run through all analysis steps and solve.

        For each step, triggers CompiledStep.solve(), advances state, and writes results to
        the Exodus output file.
        """
        file = ExodusFile(self)
        parent: CompiledStep | None = None
        for i, step in enumerate(self.steps):
            cstep = step.compile(self.model, parent=parent)
            self.dofs[1], self.flux[1] = cstep.solve(
                self.model.assemble, self.dofs[0], args=(self.ebdata,)
            )
            self.advance_state()
            file.update(i + 1, cstep.start, cstep.period, self.ebdata, self.dofs[1], self.flux[1])
            parent = cstep
            self.csteps.append(cstep)


class ExodusFile:
    """
    Wrapper for Exodus II file writing.

    Handles initialization, element block definitions, field variable
    definitions, and writing of results for each time step.
    """

    def __init__(self, simulation: "Simulation") -> None:
        """
        Create and initialize the Exodus file.

        Args:
            model: Model object to write output for.
        """
        model = simulation.model
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
        for name, ss in model.sidesets.items():
            file.put_side_set_param(sideset_id, len(ss), 0)
            file.put_side_set_name(sideset_id, str32(name))
            gids = [model.elem_map[_[0]] for _ in ss]
            sides = [_[1] + 1 for _ in ss]
            file.put_side_set_sides(sideset_id, gids, sides)
            sideset_id += 1

        # Setup result variables
        file.put_global_variable_params(1)
        file.put_global_variable_names(["time_step"])
        file.put_time(1, 0.0)
        file.put_global_variable_values(1, np.zeros(1, dtype=float))

        self.umask = np.isin(model.dof_types, [constants.Ux, constants.Uy, constants.Uz])
        self.tmask = model.dof_types == constants.T

        node_variable_params: list[str] = []
        ndim = model.coords.shape[1]
        for dim in "xyz"[:ndim]:
            node_variable_params.append(f"displ{dim}")
        if np.any(self.umask):
            for dim in "xyz"[:ndim]:
                node_variable_params.append(f"F{dim}")
        if np.any(self.tmask):
            node_variable_params.append("T")
            node_variable_params.append("HFL")
        file.put_node_variable_params(len(node_variable_params))
        file.put_node_variable_names(node_variable_params)

        zero = np.zeros(model.coords.shape[0])
        for dim in "xyz"[:ndim]:
            file.put_node_variable_values(1, f"displ{dim}", zero)
        if np.any(self.umask):
            for dim in "xyz"[:ndim]:
                file.put_node_variable_values(1, f"F{dim}", zero)

        if np.any(self.tmask):
            file.put_node_variable_values(1, "T", zero)
            file.put_node_variable_values(1, "HFL", zero)

        # Write initial element variable values
        for j, ebd in enumerate(simulation.ebdata):
            for name, value in ebd.items(projection="centroid"):
                file.put_element_variable_values(1, j + 1, name, value)

        self.file = file
        self.model = model

    def update(
        self,
        step_no: int,
        start: float,
        period: float,
        ebdata: list[ElementBlockData],
        dofs: NDArray,
        flux: NDArray,
    ) -> None:
        """
        Write updated values for a new time step.

        Args:
            step_no: Index of current step.
        """
        file = self.file
        model = self.model

        file.put_time(step_no + 1, start + period)

        ndim = model.coords.shape[1]
        u = dofs[self.umask]
        if not u.size:
            u = np.zeros(model.coords.size)
        for i in range(ndim):
            dim = "xyz"[i]
            file.put_node_variable_values(step_no + 1, f"displ{dim}", u[i::ndim])
        if np.any(self.umask):
            R = flux[self.umask]
            for i in range(ndim):
                dim = "xyz"[i]
                file.put_node_variable_values(step_no + 1, f"F{dim}", R[i::ndim])
        if np.any(self.tmask):
            file.put_node_variable_values(step_no + 1, "T", dofs[self.tmask])
            file.put_node_variable_values(step_no + 1, "HFL", flux[self.tmask])
        for j, bd in enumerate(ebdata):
            for name, value in bd.items(projection="centroid"):
                file.put_element_variable_values(step_no + 1, j + 1, name, value)


def str32(string: str) -> str:
    """
    Format string to 32 characters for Exodus set names.

    Pads or truncates to ensure 32-character width.
    """
    return f"{string:32}"
