import argparse
import sys
from typing import Sequence

import numpy as np

import fem


def exercise(esize: float = 0.05):
    class Everywhere(fem.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            return True

    class Top(fem.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            if on_boundary and x[1] > 0.999:
                return True
            return False

    class Bottom(fem.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            if on_boundary and x[1] < -0.999:
                return True
            return False

    nodes, elements = fem.meshing.plate_with_hole(esize=esize)
    mesh_builder = fem.builder.MeshBuilder(nodes=nodes, elements=elements)
    mesh_builder.block(name="Block-1", region=Everywhere(), cell_type=fem.cell.Tri3)
    mesh_builder.nodeset("Top", region=Top())
    mesh_builder.sideset("Bottom", region=Bottom())
    mesh_builder.elemset("All", region=Everywhere())
    mesh = mesh_builder.emit_mesh()

    builder = fem.builder.ModelBuilder(mesh)
    builder.assign_properties(
        block="Block-1",
        element=fem.element.CPS3(),
        material=fem.material.LinearElastic(
            density=2400.0, youngs_modulus=30.0e9, poissons_ratio=0.3
        ),
    )
    builder.boundary(nodeset="Top", bc=fem.collections.PinnedBoundary())
    builder.traction(sideset="Bottom", vector=fem.collections.ConstantVector([400e3, -300e3]))
    builder.gravity(elemset="All", vector=fem.collections.ConstantVector([0, -9.81]))
    model = builder.emit_model()

    solver = fem.solver.DirectSolver()
    solution = solver(model)
    u = solution.dofs
    U = np.linalg.norm(u, axis=1)
    print(np.amax(U))

    solver = fem.solver.NonlinearNewtonSolver()
    solution = solver(model)
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
