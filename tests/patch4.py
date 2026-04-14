import numpy as np

import felib


def test_patch4():
    X, Y = felib.X, felib.Y
    nodes = [
        [1, 0.0, 0.0],
        [2, 4.0, 0.0],
        [3, 10.0, 0.0],
        [4, 0.0, 4.50],
        [5, 5.5, 5.50],
        [6, 10.0, 5.0],
        [7, 0.0, 10.0],
        [8, 4.200000000000000178, 10.0],
        [9, 10.0, 10.0],
    ]
    elements = [[1, 1, 2, 5, 4], [2, 2, 3, 6, 5], [3, 4, 5, 8, 7], [4, 5, 6, 9, 8]]
    mesh = felib.mesh.Mesh(nodes, elements)
    mesh.block(name="All", elements=[1, 2, 3, 4], cell_type=felib.element.Quad4)
    mesh.sideset(name="IHI", sides=[[2, 2], [4, 2]])
    mesh.nodeset(name="ILO", nodes=[1, 4, 7])
    mesh.nodeset(name="FIX-Y", nodes=[1])
    mesh.nodeset(name="IHI", nodes=[3, 6, 9])

    model = felib.model.Model(mesh, name="patch4")
    m = felib.material.LinearElastic(density=1.0, youngs_modulus=1.0e03, poissons_ratio=2.5e-01)
    model.assign_properties(block="All", element=felib.element.CPE4NL(), material=m)

    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step()
    step.boundary(nodes="ILO", dofs=[X])
    step.boundary(nodes="FIX-Y", dofs=[Y])
    step.traction(sideset="IHI", magnitude=1.0, direction=[1.0, 0])

    simulation.run()
    for ebd in simulation.ebdata:
        assert np.allclose(ebd.data[0, :, :, 4], 1.0, atol = 1e-3)
