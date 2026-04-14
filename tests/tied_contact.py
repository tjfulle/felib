import numpy as np

import felib


def test_tied_mirrored():
    """Mirrored 5x2 CPS4 tied mesh matches a continuous solid mesh."""

    E = 100.0
    nu = 0.3
    axial_force = 10.0

    primary_nodes, primary_elems = felib.meshing.rectmesh(
        (0.0, 5.0, 0.0, 2.0),
        nx=5,
        ny=2,
    )
    tied = felib.meshing.mirrored_tied_mesh(primary_nodes, primary_elems, "+x")

    mesh_tied = felib.mesh.Mesh(nodes=tied.nodes, elements=tied.elements)
    mesh_tied.block(name="Primary", cell_type=felib.element.CPS4, elements=tied.primary_elements)
    mesh_tied.block(
        name="Secondary",
        cell_type=felib.element.CPS4,
        elements=tied.secondary_elements,
    )
    mesh_tied.freeze()
    assert mesh_tied.tied_node_pairs == tied.node_pairs

    material = felib.material.LinearElastic(youngs_modulus=E, poissons_ratio=nu)
    model_tied = felib.model.Model(mesh_tied, name="tied_mirrored")
    model_tied.assign_properties(block="Primary", element=felib.element.CPS4(), material=material)
    model_tied.assign_properties(
        block="Secondary",
        element=felib.element.CPS4(),
        material=material,
    )

    left_nodes = [node[0] for node in tied.nodes if np.isclose(node[1], 0.0)]
    right_nodes = [node[0] for node in tied.nodes if np.isclose(node[1], 10.0)]

    sim_tied = felib.simulation.Simulation(model_tied)
    step_tied = sim_tied.static_step()
    step_tied.boundary(nodes=left_nodes, dofs=[felib.X, felib.Y], value=0.0)
    step_tied.point_load(
        nodes=right_nodes,
        dofs=felib.X,
        value=axial_force / len(right_nodes),
    )
    sim_tied.run()
    u_tied = sim_tied.ndata.gather("u")

    for secondary, primary in tied.node_pairs:
        np.testing.assert_allclose(
            u_tied[mesh_tied.node_map.local(secondary)],
            u_tied[mesh_tied.node_map.local(primary)],
            atol=1e-12,
        )

    solid_nodes, solid_elems = felib.meshing.rectmesh(
        (0.0, 10.0, 0.0, 2.0),
        nx=10,
        ny=2,
    )
    mesh_solid = felib.mesh.Mesh(nodes=solid_nodes, elements=solid_elems)
    mesh_solid.block(
        name="Solid",
        cell_type=felib.element.CPS4,
        elements=[elem[0] for elem in solid_elems],
    )

    model_solid = felib.model.Model(mesh_solid, name="solid_mirrored_reference")
    model_solid.assign_properties(block="Solid", element=felib.element.CPS4(), material=material)

    solid_left_nodes = [node[0] for node in solid_nodes if np.isclose(node[1], 0.0)]
    solid_right_nodes = [node[0] for node in solid_nodes if np.isclose(node[1], 10.0)]

    sim_solid = felib.simulation.Simulation(model_solid)
    step_solid = sim_solid.static_step()
    step_solid.boundary(nodes=solid_left_nodes, dofs=[felib.X, felib.Y], value=0.0)
    step_solid.point_load(
        nodes=solid_right_nodes,
        dofs=felib.X,
        value=axial_force / len(solid_right_nodes),
    )
    sim_solid.run()
    u_solid = sim_solid.ndata.gather("u")

    tied_by_coord = _displacements_by_coord(mesh_tied, u_tied)
    solid_by_coord = _displacements_by_coord(mesh_solid, u_solid)

    assert tied_by_coord.keys() == solid_by_coord.keys()
    for coord in solid_by_coord:
        np.testing.assert_allclose(tied_by_coord[coord], solid_by_coord[coord], atol=1e-8)


def test_tied_interface_matches_continuous():
    """Two adjacent blocks tied at their interface match a continuous mesh."""

    E = 1.0
    nu = 0.3
    fy = -1e3
    fx = 5.0

    # ---------------------------
    # Case A: two blocks, interface tied with equations
    # ---------------------------
    nodes_tied = [
        [1, 0.0, 0.0],
        [2, 1.0, 0.0],
        [3, 1.0, 1.0],
        [4, 0.0, 1.0],
        [5, 1.0, 0.0],
        [6, 2.0, 0.0],
        [7, 2.0, 1.0],
        [8, 1.0, 1.0],
    ]
    elems_tied = [[1, 1, 2, 3, 4], [2, 5, 6, 7, 8]]

    mesh_tied = felib.mesh.Mesh(nodes=nodes_tied, elements=elems_tied)
    mesh_tied.block(name="Block-L", cell_type=felib.element.CPE4, elements=[1])
    mesh_tied.block(name="Block-R", cell_type=felib.element.CPE4, elements=[2])

    model_tied = felib.model.Model(mesh_tied, name="tied_interface")
    material = felib.material.LinearElastic(youngs_modulus=E, poissons_ratio=nu)
    model_tied.assign_properties(block="Block-L", element=felib.element.CPE4(), material=material)
    model_tied.assign_properties(block="Block-R", element=felib.element.CPE4(), material=material)

    sim_tied = felib.simulation.Simulation(model_tied)
    step_tied = sim_tied.static_step()
    step_tied.boundary(nodes=[1, 4], dofs=[felib.X, felib.Y], value=0.0)
    step_tied.point_load(nodes=[6, 7], dofs=felib.Y, value=fy)
    step_tied.point_load(nodes=[6, 7], dofs=felib.X, value=fx)

    # Tie interface node pairs (2 <-> 5 and 3 <-> 8)
    interface_pairs = [(2, 5), (3, 8)]
    step_tied.tie_nodes(interface_pairs, dofs=[felib.X, felib.Y])

    sim_tied.run()
    u_tied = sim_tied.ndata.gather("u")

    # Check interface enforcement
    assert np.allclose(u_tied[1], u_tied[4], atol=1e-12)
    assert np.allclose(u_tied[2], u_tied[7], atol=1e-12)

    # ---------------------------
    # Case B: continuous mesh (same geometry, no tie equations)
    # ---------------------------
    nodes_cont = [
        [1, 0.0, 0.0],
        [2, 1.0, 0.0],
        [3, 1.0, 1.0],
        [4, 0.0, 1.0],
        [5, 2.0, 0.0],
        [6, 2.0, 1.0],
    ]
    elems_cont = [[1, 1, 2, 3, 4], [2, 2, 5, 6, 3]]

    mesh_cont = felib.mesh.Mesh(nodes=nodes_cont, elements=elems_cont)
    mesh_cont.block(name="Block-Cont", cell_type=felib.element.CPE4, elements=[1, 2])

    model_cont = felib.model.Model(mesh_cont, name="continuous")
    model_cont.assign_properties(block="Block-Cont", element=felib.element.CPE4(), material=material)

    sim_cont = felib.simulation.Simulation(model_cont)
    step_cont = sim_cont.static_step()
    step_cont.boundary(nodes=[1, 4], dofs=[felib.X, felib.Y], value=0.0)
    step_cont.point_load(nodes=[5, 6], dofs=felib.Y, value=fy)
    step_cont.point_load(nodes=[5, 6], dofs=felib.X, value=fx)

    sim_cont.run()
    u_cont = sim_cont.ndata.gather("u")

    # Map tied nodes to continuous nodes via geometrical counterpart
    tied_idx_to_cont = [0, 1, 2, 3, 5, 6]
    u_tied_mapped = u_tied[tied_idx_to_cont]

    # Compare u at corresponding nodes to ensure the same response
    np.testing.assert_allclose(u_tied_mapped, u_cont, atol=1e-5, rtol=1e-6)

    # report error for transparency
    error = np.linalg.norm(u_tied_mapped - u_cont, axis=1)
    max_err = float(np.max(error))
    assert max_err < 1e-8


def test_node_to_surface_tie_interpolates_nonconforming_interface():
    """A fine secondary interface node is tied to a coarse primary edge."""

    nodes = [
        [1, 0.0, 0.0],
        [2, 1.0, 0.0],
        [3, 1.0, 1.0],
        [4, 0.0, 1.0],
        [5, 1.0, 0.0],
        [6, 2.0, 0.0],
        [7, 2.0, 0.5],
        [8, 1.0, 0.5],
        [9, 2.0, 1.0],
        [10, 1.0, 1.0],
    ]
    elements = [
        [1, 1, 2, 3, 4],
        [2, 5, 6, 7, 8],
        [3, 8, 7, 9, 10],
    ]

    mesh = felib.mesh.Mesh(nodes=nodes, elements=elements)
    mesh.block(name="Primary", cell_type=felib.element.CPS4, elements=[1])
    mesh.block(name="Secondary", cell_type=felib.element.CPS4, elements=[2, 3])
    mesh.sideset(name="PrimaryInterface", sides=[[1, 2]])

    material = felib.material.LinearElastic(youngs_modulus=100.0, poissons_ratio=0.3)
    model = felib.model.Model(mesh, name="nonconforming_tied_interface")
    model.assign_properties(block="Primary", element=felib.element.CPS4(), material=material)
    model.assign_properties(block="Secondary", element=felib.element.CPS4(), material=material)

    sim = felib.simulation.Simulation(model)
    step = sim.static_step()
    step.boundary(nodes=[1, 4], dofs=[felib.X, felib.Y], value=0.0)
    step.point_load(nodes=[6, 7, 9], dofs=felib.X, value=1.0 / 3.0)
    step.tie_nodes_to_surface(
        secondary_nodes=[5, 8, 10],
        primary_sides="PrimaryInterface",
        dofs=[felib.X, felib.Y],
        tol=1e-12,
    )

    equations = step.compile_constraints(model, sim.dof_manager)
    assert len(equations) == 6

    mid_x = sim.dof_manager.global_dof(mesh.node_map.local(8), felib.X)
    bottom_x = sim.dof_manager.global_dof(mesh.node_map.local(2), felib.X)
    top_x = sim.dof_manager.global_dof(mesh.node_map.local(3), felib.X)
    mid_eq = next(eq for eq in equations if int(eq[0]) == mid_x)
    terms = {
        int(mid_eq[i]): float(mid_eq[i + 1])
        for i in range(0, len(mid_eq) - 1, 2)
    }
    assert terms[mid_x] == 1.0
    assert np.isclose(terms[bottom_x], -0.5)
    assert np.isclose(terms[top_x], -0.5)

    sim.run()
    assert sim.dof_manager.has_mpc_transform
    assert sim.csteps[-1].solution.lagrange_multipliers.size == 0
    u = sim.ndata.gather("u")
    np.testing.assert_allclose(
        u[mesh.node_map.local(5)],
        u[mesh.node_map.local(2)],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        u[mesh.node_map.local(8)],
        0.5 * (u[mesh.node_map.local(2)] + u[mesh.node_map.local(3)]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        u[mesh.node_map.local(10)],
        u[mesh.node_map.local(3)],
        atol=1e-12,
    )


def test_model_registers_mpc_constraints():
    nodes = [
        [1, 0.0, 0.0],
        [2, 1.0, 0.0],
        [3, 1.0, 1.0],
        [4, 0.0, 1.0],
    ]
    elems = [[1, 1, 2, 3, 4]]
    mesh = felib.mesh.Mesh(nodes=nodes, elements=elems)
    mesh.block(name="Block", cell_type=felib.element.CPS4, elements=[1])
    model = felib.model.Model(mesh)
    mpc = felib.model.MPCConstraint(slave_dof=0, masters=[(1, 1.0)])

    model.add_mpc_constraint(mpc)

    assert model.constraints == [mpc]


def test_mpc_step_transform_propagates_prescribed_master():
    nodes = [
        [1, 0.0, 0.0],
        [2, 1.0, 0.0],
        [3, 1.0, 1.0],
        [4, 0.0, 1.0],
    ]
    elems = [[1, 1, 2, 3, 4]]
    mesh = felib.mesh.Mesh(nodes=nodes, elements=elems)
    mesh.block(name="Block", cell_type=felib.element.CPS4, elements=[1])
    model = felib.model.Model(mesh)
    model.assign_properties(
        block="Block",
        element=felib.element.CPS4(),
        material=felib.material.LinearElastic(youngs_modulus=100.0, poissons_ratio=0.3),
    )
    sim = felib.simulation.Simulation(model)

    slave = sim.dof_manager.global_dof(mesh.node_map.local(3), felib.X)
    master = sim.dof_manager.global_dof(mesh.node_map.local(2), felib.X)
    sim.dof_manager.build_mpc_map([[slave, 1.0, master, -1.0, 0.0]])

    transform, offset, reduced_free_dofs = sim.dof_manager.step_transform(
        [master],
        [0.25],
    )

    assert master not in reduced_free_dofs
    assert np.isclose(offset[master], 0.25)
    assert np.isclose(offset[slave], 0.25)
    assert np.allclose(transform[slave], 0.0)


def _displacements_by_coord(mesh, u):
    values = {}
    for coord, disp in zip(mesh.coords, u):
        key = tuple(np.round(coord, 12))
        values.setdefault(key, []).append(disp)
    return {key: np.mean(disps, axis=0) for key, disps in values.items()}
