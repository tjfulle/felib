import argparse
import sys

import numpy as np

import felib

X = felib.X
Y = felib.Y

def exercise(
    nx: int = 8,
    ny: int = 8,
    poissons_ratio: float = 0.499,
    traction: float = 1.0e9,
    displacement_scale: float = 1.0,
    plot_mesh: bool = True,
) -> None:
    nodes, elements = felib.meshing.rectmesh((0.0, 1.0, 0.0, 1.0), nx=nx, ny=ny)
    mesh = felib.mesh.Mesh(nodes=nodes, elements=elements)
    mesh.block("Block-1", region=lambda element: True, cell_type=felib.element.Quad4)
    mesh.nodeset(
        "Left",
        region=lambda node: bool(np.isclose(node.x[0], 0.0)),
    )
    mesh.sideset("Right", region=lambda side: bool(np.isclose(side.x[0], 1.0)))

    model = felib.model.Model(mesh, name="incompressible_CPE4")
    material = felib.material.LinearElastic(
        density=2400.0, youngs_modulus=30.0e9, poissons_ratio=poissons_ratio
    )
    model.assign_properties(block="Block-1", element=felib.element.CPE4(), material=material)

    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step(maxiter=25, atol=1e-10, rtol=1e-10)
    step.boundary(nodes="Left", dofs=[X, Y], value=0.0)
    step.traction(sideset="Right", magnitude=traction, direction=[1.0, 0.0])
    simulation.run()

    u = simulation.ndata["u"]
    react = simulation.csteps[-1].solution.react
    react_nodal = react.reshape((-1, 2))
    fixed_reaction = react_nodal[model.nodesets["Left"]].sum(axis=0)
    react_mag = np.linalg.norm(react_nodal, axis=1)

    print("element: CPE4")
    print(f"nu: {poissons_ratio:.3f}")
    print(f"prescribed right traction: {traction:.6e}")
    print(f"max |ux|: {np.max(np.abs(u[:, 0])):.6e}")
    print(f"max |uy|: {np.max(np.abs(u[:, 1])):.6e}")
    print(f"fixed-end reaction x: {fixed_reaction[0]:.6e}")
    print(f"fixed-end reaction y: {fixed_reaction[1]:.6e}")
    print(f"displacement plot scale: {displacement_scale:.3f}")

    if plot_mesh:
        felib.plotting.nodal_heatmap(
            model.coords + displacement_scale * u,
            model.connect,
            react_mag,
            title=f"CPE4 deformed shape with reaction magnitude heat map, scale={displacement_scale:g}",
            label="Reaction magnitude",
        )


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
        "--traction",
        type=float,
        default=1.0e9,
        help="Applied traction magnitude on the right edge [default: %(default)s]",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Deformation scale factor used for plotting [default: %(default)s]",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display the mesh plot",
    )
    args = parser.parse_args()
    exercise(
        nx=args.nx,
        ny=args.ny,
        poissons_ratio=args.nu,
        traction=args.traction,
        displacement_scale=args.scale,
        plot_mesh=not args.no_plot,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
