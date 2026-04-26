import matplotlib.pyplot as plt
import numpy as np

import felib

X, Y = felib.X, felib.Y


def build_cantilever_mesh(nelx=30, nely=4, L=10.0, H=1.0):
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
    mesh.block(
        name="All",
        elements=list(range(1, len(elements) + 1)),
        cell_type=felib.element.Quad4,
    )

    left_nodes = [j * (nelx + 1) + 1 for j in range(nely + 1)]
    right_nodes = [j * (nelx + 1) + (nelx + 1) for j in range(nely + 1)]

    mesh.nodeset(name="LEFT", nodes=left_nodes)
    mesh.nodeset(name="RIGHT", nodes=right_nodes)
    mesh.sideset(name="RIGHT_SIDE", sides=right_sides)

    return mesh, nodes, elements, left_nodes, right_nodes, L, H


def run_case(
    element_obj,
    traction_mag=8.0e2,
    youngs_modulus=1.0e3,
    poissons_ratio=0.0,
    nelx=30,
    nely=4,
    L=10.0,
    H=1.0,
):
    mesh, _, _, _, _, _, _ = build_cantilever_mesh(nelx=nelx, nely=nely, L=L, H=H)
    model = felib.model.Model(mesh, name=f"cantilever_{element_obj.__class__.__name__}")

    material = felib.material.LinearElastic(
        density=1.0,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
    )

    model.assign_properties(block="All", element=element_obj, material=material)

    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step()

    step.boundary(nodes="LEFT", dofs=[X, Y])
    step.traction(sideset="RIGHT_SIDE", magnitude=traction_mag, direction=[0.0, -1.0])

    simulation.run()
    return simulation


def final_nodal_displacements(simulation):
    data = np.asarray(simulation.ndata.data)

    if data.ndim == 3:
        return data[-1, :, :2]
    if data.ndim == 2:
        return data[:, :2]

    raise ValueError(f"Unexpected ndata.data shape: {data.shape}")


def node_xy_map(nodes):
    return {nid: np.array([x, y], dtype=float) for nid, x, y in nodes}


def plot_deformed_mesh(
    ax,
    nodes,
    elements,
    disp,
    label,
    color,
    linestyle="-",
    linewidth=1.5,
    scale=1.0,
):
    xy0 = node_xy_map(nodes)

    first = True
    for _, n1, n2, n3, n4 in elements:
        conn = [n1, n2, n3, n4, n1]
        pts = []
        for nid in conn:
            x, y = xy0[nid]
            ux, uy = disp[nid - 1, :2]
            pts.append([x + scale * ux, y + scale * uy])

        pts = np.array(pts)
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            linestyle=linestyle,
            color=color,
            linewidth=linewidth,
            label=label if first else None,
        )
        first = False


def main():
    traction = 8.0e2
    youngs_modulus = 1.0e3
    poissons_ratio = 0.0

    # 🔑 This fixes your figure
    plot_scale = 0.002

    nelx = 30
    nely = 4
    L = 10.0
    H = 1.0

    mesh, nodes, elements, left_nodes, right_nodes, L, H = build_cantilever_mesh(
        nelx=nelx, nely=nely, L=L, H=H
    )

    sim_linear = run_case(
        felib.element.CPE4(),
        traction_mag=traction,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
        nelx=nelx,
        nely=nely,
        L=L,
        H=H,
    )

    sim_nonlinear = run_case(
        felib.element.CPE4NL(),
        traction_mag=traction,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
        nelx=nelx,
        nely=nely,
        L=L,
        H=H,
    )

    u_linear = final_nodal_displacements(sim_linear)
    u_nonlinear = final_nodal_displacements(sim_nonlinear)

    right_idx = np.array(right_nodes) - 1

    print("Average right-edge displacement:")
    print(
        f"  CPE4   : ux = {u_linear[right_idx, 0].mean():.6f}, uy = {u_linear[right_idx, 1].mean():.6f}"
    )
    print(
        f"  CPE4NL : ux = {u_nonlinear[right_idx, 0].mean():.6f}, uy = {u_nonlinear[right_idx, 1].mean():.6f}"
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)

    plot_deformed_mesh(
        ax,
        nodes,
        elements,
        np.zeros_like(u_linear),
        "Undeformed",
        "0.7",
        linestyle="--",
        linewidth=1.0,
        scale=1.0,
    )

    plot_deformed_mesh(
        ax,
        nodes,
        elements,
        u_linear,
        "Linear element (CPE4)",
        "tab:blue",
        linewidth=1.8,
        scale=plot_scale,
    )

    plot_deformed_mesh(
        ax,
        nodes,
        elements,
        u_nonlinear,
        "Nonlinear element (CPE4NL)",
        "tab:red",
        linewidth=1.8,
        scale=plot_scale,
    )

    # 🔑 Critical: fix axes so beam looks like a beam
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-8.0, 1.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Cantilever beam: linear vs nonlinear elements")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
