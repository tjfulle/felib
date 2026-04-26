from typing import TYPE_CHECKING

import exodusii
import numpy as np

from ..collections import ElementBlockData
from ..collections import NodeData

if TYPE_CHECKING:
    from ..simulation import Simulation


class ExodusFileWriter:
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
        file.put_coord_names(["xyz"[i] for i in range(model.coords.shape[1])])
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

        file.put_node_variable_params(len(simulation.ndata.exodus_labels))
        file.put_node_variable_names(simulation.ndata.exodus_labels)

        for name, value in simulation.ndata.items(exodus_labels=True):
            file.put_node_variable_values(1, name, value)

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
        ndata: NodeData,
        ebdata: list[ElementBlockData],
    ) -> None:
        """
        Write updated values for a new time step.

        Args:
            step_no: Index of current step.
        """
        file = self.file
        file.put_time(step_no + 1, start + period)
        for name, value in ndata.items(exodus_labels=True):
            file.put_node_variable_values(step_no + 1, name, value)
        for j, bd in enumerate(ebdata):
            for name, value in bd.items(projection="centroid"):
                file.put_element_variable_values(step_no + 1, j + 1, name, value)


def str32(string: str) -> str:
    """
    Format string to 32 characters for Exodus set names.

    Pads or truncates to ensure 32-character width.
    """
    return f"{string:32}"
