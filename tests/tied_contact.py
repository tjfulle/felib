import numpy as np

import felib


def test_tied_interface_matches_continuous():
    """Two adjacent blocks tied at their interface match a continuous mesh."""

    E = 210e9
    nu = 0.3
    fy = -1e3

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

    # Tie interface node pairs (2 <-> 5 and 3 <-> 8)
    interface_pairs = [(2, 5), (3, 8)]
    for n_secondary, n_main in interface_pairs:
        step_tied.equation(n_secondary, felib.X, 1.0, n_main, felib.X, -1.0, 0.0)
        step_tied.equation(n_secondary, felib.Y, 1.0, n_main, felib.Y, -1.0, 0.0)

    sim_tied.run()
    u_tied = sim_tied.ndata.gather("u")

    # Check interface enforcement
    assert np.allclose(u_tied[1], u_tied[4], atol=1e-8)
    assert np.allclose(u_tied[2], u_tied[7], atol=1e-8)

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
    assert max_err < 1e-4
