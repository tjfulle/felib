import numpy as np
import matplotlib.pyplot as plt

import felib

X, Y = felib.X, felib.Y


def build_cantilever_mesh(nelx=8, nely=2, L=10.0, H=1.0):
    """
    Structured 2D cantilever beam mesh on [0, L] x [0, H].
    The beam is fixed on the left edge and traction-loaded on the right edge.

    Side numbering for the traction edge follows the same convention as the
    patch test you already used: side 2 is the right edge.
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

            if i == nelx - 1:
                right_sides.append([eid, 2])

            eid += 1

    mesh = felib.mesh.Mesh(nodes, elements)
    mesh.block(name="All", elements=list(range(1, len(elements) + 1)), cell_type=felib.element.Quad4)

    left_nodes = [j * (nelx + 1) + 1 for j in range(nely + 1)]
    right_nodes = [j * (nelx + 1) + (nelx + 1) for j in range(nely + 1)]

    mesh.nodeset(name="LEFT", nodes=left_nodes)
    mesh.nodeset(name="RIGHT", nodes=right_nodes)
    mesh.sideset(name="RIGHT_SIDE", sides=right_sides)

    return mesh, nodes, elements, right_nodes, L


def run_case(element_obj, nelx, nely, traction_mag=200.0, youngs_modulus=1.0e03, poissons_ratio=0.25):
    mesh, _, _, _, _ = build_cantilever_mesh(nelx=nelx, nely=nely)

    model = felib.model.Model(mesh, name=f"cantilever_{element_obj.__class__.__name__}_{nelx}x{nely}")
    material = felib.material.LinearElastic(
        density=1.0,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
    )

    model.assign_properties(block="All", element=element_obj, material=material)

    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step()

    # Clamp the left edge
    step.boundary(nodes="LEFT", dofs=[X, Y])

    # Downward traction on the right edge
    step.traction(sideset="RIGHT_SIDE", magnitude=traction_mag, direction=[0.0, -1.0])

    simulation.run()
    return simulation


def final_nodal_displacements(simulation):
    """
    Returns the final nodal displacement array with shape (nnodes, 2).
    The nodal solution is stored in simulation.ndata.data after run().
    """
    data = np.asarray(simulation.ndata.data)

    if data.ndim == 3:
        return data[-1, :, :2]
    if data.ndim == 2:
        return data[:, :2]

    raise ValueError(f"Unexpected ndata.data shape: {data.shape}")


def right_edge_mean_uy(simulation, right_node_ids):
    u = final_nodal_displacements(simulation)
    idx = np.array(right_node_ids, dtype=int) - 1
    return float(u[idx, 1].mean())


def mesh_convergence_study():
    traction = 200.0
    youngs_modulus = 1.0e03
    poissons_ratio = 0.25

    mesh_sizes = [4, 8, 16, 32]
    lin_vals = []
    nl_vals = []
    hs = []

    # First pass: compute responses on each mesh
    for nelx in mesh_sizes:
        nely = max(2, nelx // 4)
        mesh, nodes, elements, right_nodes, L = build_cantilever_mesh(nelx=nelx, nely=nely)
        h = L / nelx

        sim_lin = run_case(
            felib.element.CPE4(),
            nelx,
            nely,
            traction_mag=traction,
            youngs_modulus=youngs_modulus,
            poissons_ratio=poissons_ratio,
        )
        sim_nl = run_case(
            felib.element.CPE4NL(),
            nelx,
            nely,
            traction_mag=traction,
            youngs_modulus=youngs_modulus,
            poissons_ratio=poissons_ratio,
        )

        uy_lin = right_edge_mean_uy(sim_lin, right_nodes)
        uy_nl = right_edge_mean_uy(sim_nl, right_nodes)

        hs.append(h)
        lin_vals.append(uy_lin)
        nl_vals.append(uy_nl)

        print(f"nelx={nelx:2d}, nely={nely:2d}, h={h:.4f}")
        print(f"  CPE4   right-edge mean uy = {uy_lin:.8f}")
        print(f"  CPE4NL right-edge mean uy = {uy_nl:.8f}")

    hs = np.asarray(hs, dtype=float)
    lin_vals = np.asarray(lin_vals, dtype=float)
    nl_vals = np.asarray(nl_vals, dtype=float)

    # Use the finest mesh result from each formulation as its own convergence reference
    lin_ref = lin_vals[-1]
    nl_ref = nl_vals[-1]

    lin_err = np.abs(lin_vals - lin_ref)
    nl_err = np.abs(nl_vals - nl_ref)

    print("\nReference values from finest mesh:")
    print(f"  CPE4   reference uy = {lin_ref:.8f}")
    print(f"  CPE4NL reference uy = {nl_ref:.8f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Convergence of the displacement value itself
    axes[0].plot(hs, lin_vals, "o-", label="CPE4")
    axes[0].plot(hs, nl_vals, "s-", label="CPE4NL")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Element size h")
    axes[0].set_ylabel("Mean right-edge $u_y$")
    axes[0].set_title("Cantilever response vs mesh size")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Error convergence
    axes[1].loglog(hs, lin_err, "o-", label="CPE4 error")
    axes[1].loglog(hs, nl_err, "s-", label="CPE4NL error")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("Element size h")
    axes[1].set_ylabel("Absolute error in mean $u_y$")
    axes[1].set_title("Mesh convergence")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    plt.show()


def plot_finest_mesh_comparison():
    """
    Optional: visualize the deformed cantilever on the finest mesh.
    """
    nelx = 32
    nely = 8
    traction = 200.0
    youngs_modulus = 1.0e03
    poissons_ratio = 0.25

    mesh, nodes, elements, right_nodes, L = build_cantilever_mesh(nelx=nelx, nely=nely)

    sim_lin = run_case(
        felib.element.CPE4(),
        nelx,
        nely,
        traction_mag=traction,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
    )
    sim_nl = run_case(
        felib.element.CPE4NL(),
        nelx,
        nely,
        traction_mag=traction,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
    )

    u_lin = final_nodal_displacements(sim_lin)
    u_nl = final_nodal_displacements(sim_nl)

    xy0 = {nid: np.array([x, y], dtype=float) for nid, x, y in nodes}

    def plot_mesh(ax, disp, label, color, linestyle="-"):
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
                linewidth=1.0,
                label=label if first else None,
            )
            first = False

    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5), constrained_layout=True)
    plot_mesh(ax, np.zeros_like(u_lin), "Original mesh", "0.75", linestyle="--")
    plot_mesh(ax, u_lin, "CPE4", "tab:blue")
    plot_mesh(ax, u_nl, "CPE4NL", "tab:red")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Deformed x-coordinate")
    ax.set_ylabel("Deformed y-coordinate")
    ax.set_title("Finest-mesh cantilever deformation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    mesh_convergence_study()
    plot_finest_mesh_comparison()