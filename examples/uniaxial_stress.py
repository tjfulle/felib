import argparse
import sys
from typing import Sequence

import numpy as np

import fem


def exercise(esize: float = 0.05):
    class Everywhere(fem.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            return True

    class Bottom(fem.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            if on_boundary and x[1] < -0.999:
                return True
            return False

    nodes, elements = fem.meshing.plate_with_hole(esize=esize)
    mesh_builder = fem.mesh.MeshBuilder(nodes=nodes, elements=elements)
    mesh_builder.block(name="Block-1", region=Everywhere(), cell_type=fem.cell.Tri3)
    mesh_builder.nodeset("Point", region=lambda x, on_boundary: abs(x[0]) < 0.05 and x[1] > 0.999)
    mesh_builder.nodeset("Top", region=lambda x, on_boundary: x[1] > 0.99)
    mesh_builder.sideset("Bottom", region=Bottom())
    mesh = mesh_builder.build()

    builder = fem.model.ModelBuilder(mesh, name="uniaxial_stress")
    material = fem.material.LinearElastic(density=2400.0, youngs_modulus=30.0e9, poissons_ratio=0.3)
    builder.assign_properties(block="Block-1", element=fem.element.CPS3(), material=material)
    model = builder.build()

    simulation = fem.simulation.Simulation(model)
    step = simulation.static_step()
    step.boundary(nodes="Point", dofs=[0, 1], value=0.0)
    step.boundary(nodes="Top", dofs=[1], value=0.0)
    step.traction(sideset="Bottom", magnitude=1e8, direction=[0, -1])
    simulation.run()

    solution = simulation.csteps[-1].solution

    u = solution.dofs
    U = np.linalg.norm(u, axis=1)
    print(np.amax(U))
    print(solution.iterations)

    scale = 0.25 / np.max(np.abs(u))
    fem.plotting.tplot(model.coords + scale * u, model.connect, U)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-s", type=float, default=0.05, help="Element size [default: %(default)s]")
    args = p.parse_args()
    exercise(esize=args.s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
