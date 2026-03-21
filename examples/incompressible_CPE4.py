import argparse
import sys

import numpy as np

import felib


def exercise(nx: int = 8, ny: int = 8, poissons_ratio: float = 0.499, uy: float = -0.05):
    nodes, elements = felib.meshing.rectmesh((0.0, 1.0, 0.0, 1.0), nx=nx, ny=ny)
    mesh = felib.mesh.Mesh(nodes=nodes, elements=elements)
    mesh.block("Block-1", region=lambda element: True, cell_type=felib.element.Quad4)
    mesh.nodeset("Bottom", region=lambda node: bool(np.isclose(node.x[1], 0.0)))
    mesh.nodeset("Top", region=lambda node: bool(np.isclose(node.x[1], 1.0)))
    mesh.nodeset(
        "Bottom Left",
        region=lambda node: bool(np.isclose(node.x[0], 0.0) and np.isclose(node.x[1], 0.0)),
    )

    model = felib.model.Model(mesh, name="incompressible_CPE4")
    material = felib.material.LinearElastic(
        density=1.0, youngs_modulus=1000.0, poissons_ratio=poissons_ratio
    )
    model.assign_properties(block="Block-1", element=felib.element.CPE4(), material=material)

    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step(maxiter=25, atol=1e-10, rtol=1e-10)
    step.boundary(nodes="Bottom", dofs=[felib.Y], value=0.0)
    step.boundary(nodes="Bottom Left", dofs=[felib.X], value=0.0)
    step.boundary(nodes="Top", dofs=[felib.Y], value=uy)
    simulation.run()

    u = simulation.ndata["u"]
    react = simulation.csteps[-1].solution.react
    top_reaction = sum(
        react[simulation.dof_manager.global_dof(node, felib.Y)] for node in model.nodesets["Top"]
    )

    print("element: CPE4")
    print("nu: {poissons_ratio:.3f}")
    print("prescribed top uy: {uy:.6f}")
    print("max |ux|: {np.max(np.abs(u[:, 0])):.6e}")
    print("total top reaction y: {top_reaction:.6e}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=8, help="Number of elements in x [default: %(default)s]")
    parser.add_argument("--ny", type=int, default=8, help="Number of elements in y [default: %(default)s]")
    parser.add_argument(
        "--nu",
        type=float,
        default=0.499,
        help="Poisson's ratio [default: %(default)s]",
    )
    parser.add_argument(
        "--uy",
        type=float,
        default=-0.05,
        help="Prescribed top displacement [default: %(default)s]",
    )
    args = parser.parse_args()
    exercise(nx=args.nx, ny=args.ny, poissons_ratio=args.nu, uy=args.uy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
#Code generated with CODEX from ChatGPT, adapted for ME7540