import argparse
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

import felib

R = felib.X
Z = felib.Y

def lame_constants(ri: float, ro: float, pi: float, po: float = 0.0) -> tuple[float, float]:
    a2 = ri**2
    b2 = ro**2
    A = (pi * a2 - po * b2) / (b2 - a2)
    B = a2 * b2 * (pi - po) / (b2 - a2)
    return A, B


def radial_displacement_exact(
    r: np.ndarray, ri: float, ro: float, pi: float, E: float, nu: float, po: float = 0.0
) -> np.ndarray:
    A, B = lame_constants(ri, ro, pi, po)
    return (1.0 + nu) / E * ((1.0 - 2.0 * nu) * A * r + B / r)


def hoop_stress_exact(
    r: np.ndarray, ri: float, ro: float, pi: float, po: float = 0.0
) -> np.ndarray:
    A, B = lame_constants(ri, ro, pi, po)
    return A + B / r**2


def exercise(
    ri: float = 0.05,
    ro: float = 0.10,
    length: float = 0.02,
    pressure: float = 100.0e6,
    nr: int = 16,
    nz: int = 8,
):
    class Everywhere(felib.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            return True

    class InnerRadius(felib.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            return on_boundary and np.isclose(x[0], ri)

    class Bottom(felib.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            return on_boundary and np.isclose(x[1], 0.0)

    class Top(felib.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            return on_boundary and np.isclose(x[1], length)

    nodes, elements = felib.meshing.rectmesh((ri, ro, 0.0, length), nx=nr, ny=nz)
    mesh = felib.mesh.Mesh(nodes=nodes, elements=elements)
    mesh.block(name="Cylinder", region=Everywhere(), cell_type=felib.element.Quad4)
    mesh.sideset("Inner", region=InnerRadius())
    mesh.nodeset("Bottom", region=Bottom())
    mesh.nodeset("Top", region=Top())

    material = felib.material.LinearElastic(
        density=7800.0,
        youngs_modulus=210.0e9,
        poissons_ratio=0.3,
    )
    model = felib.model.Model(mesh, name="Axisymmetric Thick Cylinder")
    model.assign_properties(block="Cylinder", element=felib.element.CAX4(), material=material)

    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step()
    step.boundary(nodes="Bottom", dofs=[Z], value=0.0)
    step.boundary(nodes="Top", dofs=[Z], value=0.0)
    step.pressure(sideset="Inner", magnitude=pressure)

    simulation.run()
    solution = simulation.csteps[0].solution

    u = solution.dofs.reshape((model.nnode, -1))
    ur = u[:, R]
    uz = u[:, Z]
    print(f"Newton iterations: {solution.iterations}")
    print(f"Max radial displacement: {np.max(ur):.6e} m")
    print(f"Min radial displacement: {np.min(ur):.6e} m")
    print(f"Max axial displacement: {np.max(np.abs(uz)):.6e} m")

    scale = 0.15 * (ro - ri) / max(np.max(np.abs(u)), 1e-16)
    fig, ax = felib.plotting.mesh_plot_quad4(
    model.coords,
    model.connect,
    color="0.6",
    label="Original mesh"
)

    # Count existing lines (original mesh)
    n_before = len(ax.lines)

    # Plot deformed mesh
    felib.plotting.mesh_plot_quad4(
        model.coords + scale * u,
        model.connect,
        ax=ax,
        color="tab:red",
        label="Deformed mesh"
    )

    # Count new lines (deformed mesh)
    n_after = len(ax.lines)

    # Apply dashed style ONLY to deformed mesh
    for line in ax.lines[n_before:n_after]:
        line.set_linestyle("--")
        line.set_linewidth(1.5)

    # Optional: slightly soften original mesh
    for line in ax.lines[:n_before]:
        line.set_linewidth(1.0)
        line.set_alpha(0.8)

    # Formatting
    ax.set_aspect("equal")
    ax.set_xlabel("r")
    ax.set_ylabel("z")
    ax.set_title("Axisymmetric Thick Cylinder")

    # Legend at top-right
    ax.legend(loc="upper right")

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.show()
    
    tris = np.zeros((2 * len(model.connect), 3), dtype=int)
    for i, q in enumerate(model.connect):
        tris[2 * i] = [q[0], q[1], q[2]]
        tris[2 * i + 1] = [q[0], q[2], q[3]]
    felib.plotting.tplot(model.coords, tris, ur,            title="Axisymmetric Thick Cylinder: Radial Displacement")
    
    # Build nodal hoop stress from element integration-point history for comparison.
    block = model.blocks[0]
    var_names = block.element_variable_names()
    hoop_idx = var_names.index("szz")
    elem_hoop = block.pdata[1, :, :, hoop_idx].mean(axis=1)
    nodal_hoop = np.zeros(model.nnode, dtype=float)
    nodal_count = np.zeros(model.nnode, dtype=float)
    for e, conn in enumerate(model.connect):
        nodal_hoop[conn] += elem_hoop[e]
        nodal_count[conn] += 1.0
    nodal_hoop /= np.maximum(nodal_count, 1.0)

    # Average FE results over nodes with the same radius to compare against Lamé theory.
    radii = model.coords[:, R]
    unique_r = np.unique(np.round(radii, decimals=12))
    fe_ur = np.zeros_like(unique_r)
    fe_hoop = np.zeros_like(unique_r)
    for i, r in enumerate(unique_r):
        mask = np.isclose(radii, r)
        fe_ur[i] = np.mean(ur[mask])
        fe_hoop[i] = np.mean(nodal_hoop[mask])

    youngs_modulus = material.youngs_modulus
    poissons_ratio = material.poissons_ratio
    exact_ur = radial_displacement_exact(
        unique_r, ri, ro, pressure, youngs_modulus, poissons_ratio
    )
    exact_hoop = hoop_stress_exact(unique_r, ri, ro, pressure)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(unique_r, exact_ur, "-", color="k", label="Analytical")
    axs[0].plot(unique_r, fe_ur, "o", color="tab:blue", label="FE")
    axs[0].set_xlabel("r [m]")
    axs[0].set_ylabel("u_r [m]")
    axs[0].set_title("Radial Displacement")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(unique_r, exact_hoop, "-", color="k", label="Analytical")
    axs[1].plot(unique_r, fe_hoop, "o", color="tab:red", label="FE")
    axs[1].set_xlabel("r [m]")
    axs[1].set_ylabel(r"$\sigma_\theta$ [Pa]")
    axs[1].set_title("Hoop Stress")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ri", type=float, default=0.05, help="Inner radius [m]")
    p.add_argument("--ro", type=float, default=0.10, help="Outer radius [m]")
    p.add_argument("--length", type=float, default=0.02, help="Axial length of rz slice [m]")
    p.add_argument("--pressure", type=float, default=100.0e6, help="Internal pressure [Pa]")
    p.add_argument("--nr", type=int, default=16, help="Number of elements through thickness")
    p.add_argument("--nz", type=int, default=8, help="Number of elements in axial direction")
    args = p.parse_args()
    exercise(
        ri=args.ri,
        ro=args.ro,
        length=args.length,
        pressure=args.pressure,
        nr=args.nr,
        nz=args.nz,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
