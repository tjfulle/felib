import matplotlib.pyplot as plt
import numpy as np

import felib

X, Y = felib.X, felib.Y


def build_bar_mesh(nelx=10, nely=3, L=10.0, H=1.0):
    """
    Slender 2D bar mesh on [0, L] x [0, H].
    """
    nodes = []
    nid = 1
    for j in range(nely + 1):
        y = H * j / nely
        for i in range(nelx + 1):
            x = L * i / nelx
            nodes.append([nid, x, y])
            nid += 1

    elements = []
    right_sides = []

    eid = 1
    for j in range(nely):
        for i in range(nelx):
            n1 = j * (nelx + 1) + i + 1
            n2 = n1 + 1
            n4 = n1 + (nelx + 1)
            n3 = n4 + 1
            elements.append([eid, n1, n2, n3, n4])

            # Side 2 is the right edge in the element ordering used here.
            if i == nelx - 1:
                right_sides.append([eid, 2])

            eid += 1

    mesh = felib.mesh.Mesh(nodes, elements)
    mesh.block(
        name="All", elements=list(range(1, len(elements) + 1)), cell_type=felib.element.Quad4
    )

    left_nodes = [j * (nelx + 1) + 1 for j in range(nely + 1)]
    right_nodes = [j * (nelx + 1) + (nelx + 1) for j in range(nely + 1)]
    bottom_nodes = list(range(1, nelx + 2))

    mesh.nodeset(name="LEFT", nodes=left_nodes)
    mesh.nodeset(name="RIGHT", nodes=right_nodes)
    mesh.nodeset(name="BOTTOM", nodes=bottom_nodes)
    mesh.sideset(name="RIGHT_SIDE", sides=right_sides)

    return mesh, nodes, elements, right_nodes, L


def run_case(element_obj, traction_mag=750.0, youngs_modulus=1.0e03, poissons_ratio=0.0):
    mesh, _, _, _, _ = build_bar_mesh()
    model = felib.model.Model(mesh, name=f"bar_{element_obj.__class__.__name__}")

    material = felib.material.LinearElastic(
        density=1.0,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
    )

    model.assign_properties(block="All", element=element_obj, material=material)

    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step()

    # Keep the problem well-posed.
    step.boundary(nodes="LEFT", dofs=[X])
    step.boundary(nodes="BOTTOM", dofs=[Y])

    # Traction on the right side.
    step.traction(sideset="RIGHT_SIDE", magnitude=traction_mag, direction=[1.0, 0.0])

    simulation.run()
    return simulation


def final_nodal_displacements(simulation):
    """
    simulation.ndata.data is populated by scatter_dofs(u).
    The final state is the last index.
    """
    data = np.asarray(simulation.ndata.data)

    if data.ndim == 3:
        return data[-1, :, :2]
    if data.ndim == 2:
        return data[:, :2]

    raise ValueError(f"Unexpected ndata.data shape: {data.shape}")


def node_xy_map(nodes):
    return {nid: np.array([x, y], dtype=float) for nid, x, y in nodes}


def finite_strain_bar_reference(traction, length, youngs_modulus):
    """
    1D finite-strain reference for a linear elastic material:
        T = (E / 2) * (lambda^3 - lambda)

    Solve for lambda > 0.
    """
    coeffs = [1.0, 0.0, -1.0, -(2.0 * traction / youngs_modulus)]
    roots = np.roots(coeffs)
    roots = roots[np.isclose(roots.imag, 0.0)].real
    roots = roots[roots > 0.0]
    if roots.size == 0:
        raise ValueError("No positive real stretch ratio found.")
    return roots[np.argmin(np.abs(roots - 1.0))]


def analytical_displacement_field(nodes, traction, length, youngs_modulus):
    """
    Affine stretch reference mapped onto the 2D bar:
        u_x = (lambda - 1) * x
        u_y = 0
    """
    lam = finite_strain_bar_reference(traction, length, youngs_modulus)
    xy = np.array([[x, y] for _, x, y in nodes], dtype=float)
    disp = np.zeros((len(nodes), 2), dtype=float)
    disp[:, 0] = (lam - 1.0) * xy[:, 0]
    disp[:, 1] = 0.0
    return disp, lam


def plot_deformed_mesh(ax, nodes, elements, disp, label, color, linestyle="-"):
    xy0 = node_xy_map(nodes)

    first = True
    for _, n1, n2, n3, n4 in elements:
        conn = [n1, n2, n3, n4, n1]
        pts = []
        for nid in conn:
            x, y = xy0[nid]
            ux, uy = disp[nid - 1, :2]
            pts.append([x + ux, y + uy])

        pts = np.array(pts)
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            linestyle=linestyle,
            color=color,
            linewidth=1.5,
            label=label if first else None,
        )
        first = False


def main():
    traction = 750.0
    youngs_modulus = 1.0e03
    poissons_ratio = 0.0

    mesh, nodes, elements, right_nodes, L = build_bar_mesh()

    sim_linear = run_case(
        felib.element.CPE4(),
        traction_mag=traction,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
    )
    sim_nonlinear = run_case(
        felib.element.CPE4NL(),
        traction_mag=traction,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
    )

    u_linear = final_nodal_displacements(sim_linear)
    u_nonlinear = final_nodal_displacements(sim_nonlinear)

    u_analytical, lam = analytical_displacement_field(nodes, traction, L, youngs_modulus)

    print(f"Analytical stretch ratio lambda = {lam:.6f}")
    print("Mean right-edge ux:")
    print("  CPE4   =", u_linear[np.array(right_nodes) - 1, 0].mean())
    print("  CPE4NL =", u_nonlinear[np.array(right_nodes) - 1, 0].mean())
    print("  Analytical =", u_analytical[np.array(right_nodes) - 1, 0].mean())

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), constrained_layout=True)

    plot_deformed_mesh(
        ax, nodes, elements, np.zeros_like(u_linear), "Original mesh", "0.75", linestyle="--"
    )
    plot_deformed_mesh(ax, nodes, elements, u_linear, "CPE4", "tab:blue")
    plot_deformed_mesh(ax, nodes, elements, u_nonlinear, "CPE4NL", "tab:red")
    plot_deformed_mesh(ax, nodes, elements, u_analytical, "Analytical", "k", linestyle="--")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Deformed x-coordinate")
    ax.set_ylabel("Deformed y-coordinate")
    ax.set_title("2D bar under large traction: mesh displacement comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
