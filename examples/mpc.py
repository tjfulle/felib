import sys

import numpy as np

import felib

X = felib.X
Y = felib.Y


def mpc():
    class Everywhere(felib.collections.ElementSelector):
        def __call__(self, element: felib.collections.Element) -> bool:
            return True

    nodes = [[1, 0.0, 0.0], [2, 1.0, 0.0], [3, 1.0, 1.0], [4, 0.0, 1.0], [5, 0.5, 0.5]]
    elements = [[1, 1, 2, 5], [2, 2, 3, 5], [3, 3, 4, 5], [4, 4, 1, 5]]
    mesh = felib.mesh.Mesh(nodes=nodes, elements=elements)
    mesh.block(name="Block-1", region=Everywhere(), cell_type=felib.element.Tri3)
    mesh.nodeset("Boundary", nodes=[1, 2, 3, 4])

    m = felib.material.LinearElastic(density=2400.0, youngs_modulus=30.0e9, poissons_ratio=0.3)
    model = felib.model.Model(mesh, name="plate_with_hole")
    model.assign_properties(block="Block-1", element=felib.element.CPS3(), material=m)

    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step()
    step.boundary(nodes="Boundary", dofs=[X, Y], value=0.0)
    step.point_load(nodes=5, dofs=[Y], value=-1e3)
    step.equation(5, X, 1.0, 5, Y, -1.0, 0.0)

    simulation.run()

    u = simulation.ndata["u"]
    U = np.linalg.norm(u, axis=1)
    print(np.amax(U))

    scale = 0.25 / np.max(np.abs(u))
    felib.plotting.tplot(model.coords + scale * u, model.connect, U)


def main() -> int:
    mpc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
