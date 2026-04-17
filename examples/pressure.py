import argparse
import sys

import numpy as np

import felib


def exercise(esize: float = 0.05):
    class Everywhere(felib.collections.ElementSelector):
        def __call__(self, element: felib.collections.Element):
            return True

    class Inside(felib.collections.SideSelector):
        def __call__(self, side: felib.collections.Side):
            return abs(side.x[0]) < 0.8 and abs(side.x[1]) < 0.8

    nodes, elements = felib.meshing.plate_with_hole(esize=esize)
    mesh = felib.mesh.Mesh(nodes=nodes, elements=elements)
    mesh.block(name="Block-1", region=Everywhere(), cell_type=felib.element.Tri3)
    mesh.nodeset("Top Left", region=lambda node: node.x[0] < -0.99 and node.x[1] > 0.99)
    mesh.nodeset("Top Right", region=lambda node: node.x[0] > 0.99 and node.x[1] > 0.99)
    mesh.sideset("Inside", region=Inside())
    mesh.elemset("All", region=Everywhere())

    material = felib.material.LinearElastic(
        density=2400.0, youngs_modulus=30.0e9, poissons_ratio=0.3
    )
    model = felib.model.Model(mesh, name="Pressure")
    model.assign_properties(block="Block-1", element=felib.element.CPS3(), material=material)

    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step()
    step.boundary(nodes="Top Right", dofs=[1], value=0.0)
    step.boundary(nodes="Top Left", dofs=[0, 1], value=0.0)
    step.pressure(sideset="Inside", magnitude=500e3)

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
