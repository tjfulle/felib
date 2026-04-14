import inspect
import numpy as np
import matplotlib.pyplot as plt

import felib

X, Y = felib.X, felib.Y


def build_square_mesh():
    """
    2x2 square block on [0, 1] x [0, 1].
    """
    nodes = [
        [1, 0.0, 0.0],
        [2, 0.5, 0.0],
        [3, 1.0, 0.0],
        [4, 0.0, 0.5],
        [5, 0.5, 0.5],
        [6, 1.0, 0.5],
        [7, 0.0, 1.0],
        [8, 0.5, 1.0],
        [9, 1.0, 1.0],
    ]
    elements = [
        [1, 1, 2, 5, 4],
        [2, 2, 3, 6, 5],
        [3, 4, 5, 8, 7],
        [4, 5, 6, 9, 8],
    ]

    mesh = felib.mesh.Mesh(nodes, elements)
    mesh.block(name="All", elements=[1, 2, 3, 4], cell_type=felib.element.Quad4)

    # Convenient named sets for boundary conditions
    mesh.nodeset(name="LEFT", nodes=[1, 4, 7])
    mesh.nodeset(name="RIGHT", nodes=[3, 6, 9])
    mesh.nodeset(name="BOTTOM", nodes=[1, 2, 3])
    mesh.nodeset(name="TOP", nodes=[7, 8, 9])

    return mesh, nodes, elements


def _apply_prescribed_bc(step, *, nodes, dofs, value):
    """
    Try to apply a nonzero prescribed displacement without guessing a single API.
    If the boundary method supports a value-like keyword, use it.
    """
    sig = inspect.signature(step.boundary)
    params = sig.parameters

    kwargs = {"nodes": nodes, "dofs": dofs}

    for key in ("value", "values", "u", "displacement", "components", "prescribed", "magnitude"):
        if key in params:
            kwargs[key] = value
            step.boundary(**kwargs)
            return

    # If we get here, the method likely only supports zero-valued fixities.
    raise RuntimeError(
        "step.boundary(...) does not appear to accept a nonzero displacement keyword. "
        "Please send the step.boundary method definition and I will adapt this test exactly."
    )


def run_case(element_obj, stretch=0.50):
    mesh, _, _ = build_square_mesh()
    model = felib.model.Model(mesh, name=f"square_{element_obj.__class__.__name__}")

    material = felib.material.LinearElastic(
        density=1.0,
        youngs_modulus=1.0e03,
        poissons_ratio=2.5e-01,
    )

    model.assign_properties(block="All", element=element_obj, material=material)

    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step()

    # Prescribed stretch in x
    _apply_prescribed_bc(step, nodes="LEFT", dofs=[X, Y], value=0.0)
    _apply_prescribed_bc(step, nodes="BOTTOM", dofs=[Y], value=0.0)
    _apply_prescribed_bc(step, nodes="TOP", dofs=[Y], value=0.0)
    _apply_prescribed_bc(step, nodes="RIGHT", dofs=[X], value=stretch)

    simulation.run()
    return simulation


def get_nodal_displacements(simulation):
    """
    simulation.ndata.data is a NumPy array populated by scatter_dofs(u).
    For this test, the final state is stored in the last index.
    """
    data = np.asarray(simulation.ndata.data)

    if data.ndim == 3:
        return data[-1, :, :2]
    if data.ndim == 2:
        return data[:, :2]

    raise ValueError(f"Unexpected ndata.data shape: {data.shape}")


def analytical_ux(x, L, stretch):
    """
    Simple affine stretch reference:
        u_x = stretch * x / L
        u_y = 0
    """
    return stretch * x / L


def node_xy_map(nodes):
    return {nid: np.array([x, y], dtype=float) for nid, x, y in nodes}


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
    stretch = 0.50
    L = 1.0

    sim_linear = run_case(felib.element.CPE4(), stretch=stretch)
    sim_nonlinear = run_case(felib.element.CPE4NL(), stretch=stretch)

    mesh, nodes, elements = build_square_mesh()
    xy0 = node_xy_map(nodes)
    x_nodes = np.array([xy0[nid][0] for nid, _, _ in nodes], dtype=float)

    u_linear = get_nodal_displacements(sim_linear)
    u_nonlinear = get_nodal_displacements(sim_nonlinear)

    ux_linear = u_linear[:, 0]
    ux_nonlinear = u_nonlinear[:, 0]

    x_line = np.linspace(0.0, L, 200)
    ux_ref_line = analytical_ux(x_line, L, stretch)
    ux_ref_nodes = analytical_ux(x_nodes, L, stretch)

    print("CPE4   nodal ux =", ux_linear)
    print("CPE4NL nodal ux =", ux_nonlinear)
    print("Max |CPE4   - analytical| =", np.max(np.abs(ux_linear - ux_ref_nodes)))
    print("Max |CPE4NL - analytical| =", np.max(np.abs(ux_nonlinear - ux_ref_nodes)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    axes[0].scatter(x_nodes, ux_linear, marker="o", s=45, label="CPE4 numerical")
    axes[0].scatter(x_nodes, ux_nonlinear, marker="s", s=45, label="CPE4NL numerical")
    axes[0].plot(x_line, ux_ref_line, "k--", linewidth=2.0, label="Analytical affine stretch")
    axes[0].set_xlabel("Original x-coordinate")
    axes[0].set_ylabel("Horizontal displacement $u_x$")
    axes[0].set_title("Square block: nodal displacement comparison")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    plot_deformed_mesh(axes[1], nodes, elements, np.zeros_like(u_linear), "Original mesh", "0.75", linestyle="--")
    plot_deformed_mesh(axes[1], nodes, elements, u_linear, "CPE4", "tab:blue")
    plot_deformed_mesh(axes[1], nodes, elements, u_nonlinear, "CPE4NL", "tab:red")

    analytical_disp = np.zeros_like(u_linear)
    analytical_disp[:, 0] = ux_ref_nodes
    analytical_disp[:, 1] = 0.0
    plot_deformed_mesh(axes[1], nodes, elements, analytical_disp, "Analytical", "k", linestyle="--")

    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("Deformed x-coordinate")
    axes[1].set_ylabel("Deformed y-coordinate")
    axes[1].set_title("Square block: deformed mesh comparison")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.show()


if __name__ == "__main__":
    main()
    