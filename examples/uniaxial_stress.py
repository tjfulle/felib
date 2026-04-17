import argparse
import sys

import numpy as np

import felib


def exercise(esize: float = 0.05):
    class Everywhere(felib.collections.ElementSelector):
        def __call__(self, element: felib.collections.Element):
            return True

    class Bottom(felib.collections.SideSelector):
        def __call__(self, side: felib.collections.Side):
            return side.x[1] < -0.999

    nodes, elements = felib.meshing.plate_with_hole(esize=esize)
    mesh = felib.mesh.Mesh(nodes=nodes, elements=elements)
    mesh.block(name="Block-1", region=Everywhere(), cell_type=felib.element.Tri3)
    mesh.nodeset("Point", region=lambda node: abs(node.x[0]) < 0.05 and node.x[1] > 0.999)
    mesh.nodeset("Top", region=lambda node: node.x[1] > 0.99)
    mesh.sideset("Bottom", region=Bottom())

    model = felib.model.Model(mesh, name="uniaxial_stress")
    material = felib.material.LinearElastic(
        density=2400.0, youngs_modulus=30.0e9, poissons_ratio=0.3
    )
    model.assign_properties(block="Block-1", element=felib.element.CPS3(), material=material)

    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step()
    step.boundary(nodes="Point", dofs=[0, 1], value=0.0)
    step.boundary(nodes="Top", dofs=[1], value=0.0)
    step.traction(sideset="Bottom", magnitude=1e8, direction=[0, -1])
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
