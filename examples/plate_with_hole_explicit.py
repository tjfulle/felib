import argparse
import sys

import numpy as np

import felib
from felib import simulation

X = felib.X
Y = felib.Y


def exercise(esize: float = 0.05):
    class Everywhere(felib.collections.ElementSelector):
        def __call__(self, element: felib.collections.Element) -> bool:
            return True

    class Top(felib.collections.NodeSelector):
        def __call__(self, node: felib.collections.Node) -> bool:
            if node.on_boundary and node.x[1] > 0.999:
                return True
            return False

    class Bottom(felib.collections.SideSelector):
        def __call__(self, side: felib.collections.Side):
            return side.x[1] < -0.999

    nodes, elements = felib.meshing.plate_with_hole(esize=esize)
    mesh = felib.mesh.Mesh(nodes=nodes, elements=elements)
    mesh.block(name="Block-1", region=Everywhere(), cell_type=felib.element.Tri3)
    mesh.nodeset("Top", region=Top())
    mesh.sideset("Bottom", region=Bottom())
    mesh.elemset("All", region=Everywhere())

    m = felib.material.LinearElastic(density=2400.0, youngs_modulus=30.0e9, poissons_ratio=0.3)
    model = felib.model.Model(mesh, name="plate_with_hole")
    model.assign_properties(block="Block-1", element=felib.element.CPS3(), material=m)

    simulation = felib.simulation.Simulation(model)
    step = simulation.explicit_step(period=1.0e-4)
    step.boundary(nodes="Top", dofs=[X, Y], value=0.0)
    step.traction(sideset="Bottom", magnitude=500e3, direction=[4 / 5, -3 / 5])
    step.gravity(elements="All", g=9.81, direction=[0, -1])

    simulation.run()

    u = simulation.ndata["u"]
    U = np.linalg.norm(u, axis=1)
    print(np.amax(U))

    scale = 0.25 / np.max(np.abs(u))
    felib.plotting.tplot(model.coords + scale * u, model.connect, U)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-s", type=float, default=0.05, help="Element size [default: %(default)s]")
    args = p.parse_args()
    exercise(esize=args.s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
