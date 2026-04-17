import matplotlib.pyplot as plt
import numpy as np

import felib

X, Y = felib.X, felib.Y


def build_bar_mesh(nelx=8, nely=2, L=10.0, H=1.0):
    """
    Slender rectangular bar:
        [0, L] x [0, H]

    This is a better geometric-nonlinearity benchmark than the square affine test.
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

            # In this codebase, side 2 is the right edge, as in your patch test.
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


def run_case(element_obj, traction_mag, youngs_modulus=1.0e03, poissons_ratio=0.0):
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

    # Zero constraints to remove rigid-body motion and keep the bar-like response clean.
    step.boundary(nodes="LEFT", dofs=[X])
    step.boundary(nodes="BOTTOM", dofs=[Y])

    # Large x-traction on the right side.
    step.traction(sideset="RIGHT_SIDE", magnitude=traction_mag, direction=[1.0, 0.0])

    simulation.run()
    return simulation


def final_nodal_displacements(simulation):
    """
    simulation.ndata.data is a NumPy array populated by scatter_dofs(u).
    We use the final state.
    """
    data = np.asarray(simulation.ndata.data)
    if data.ndim == 3:
        return data[-1, :, :2]
    if data.ndim == 2:
        return data[:, :2]
    raise ValueError(f"Unexpected ndata.data shape: {data.shape}")


def right_edge_mean_ux(simulation, right_node_ids):
    u = final_nodal_displacements(simulation)
    idx = np.array(right_node_ids, dtype=int) - 1
    return float(u[idx, 0].mean())


def plot_displacement_along_bar(traction=750.0):
    youngs_modulus = 1.0e03
    poissons_ratio = 0.0

    mesh, nodes, _, _, L = build_bar_mesh()

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

    u_lin = final_nodal_displacements(sim_linear)
    u_nl = final_nodal_displacements(sim_nonlinear)

    # Extract x-coordinates
    x = np.array([node[1] for node in nodes])

    # Sort for clean plotting
    order = np.argsort(x)
    x_sorted = x[order]
    ux_lin = u_lin[order, 0]
    ux_nl = u_nl[order, 0]

    # Analytical curve
    x_line = np.linspace(0.0, L, 200)
    lam = np.roots([1, 0, -1, -(2 * traction / youngs_modulus)])
    lam = lam[np.isreal(lam)].real
    lam = lam[lam > 0][0]
    ux_ref = (lam - 1.0) * x_line

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(x_sorted, ux_lin, "o-", label="CPE4")
    plt.plot(x_sorted, ux_nl, "s-", label="CPE4NL")
    plt.plot(x_line, ux_ref, "k--", linewidth=2, label="Analytical")

    plt.xlabel("x-position along bar")
    plt.ylabel("Horizontal displacement $u_x$")
    plt.title(f"Displacement along bar (traction = {traction})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def finite_strain_bar_reference(traction, length, youngs_modulus):
    """
    1D finite-strain bar reference for a linear elastic material in PK2 form:
        T = (E/2) * (lambda^3 - lambda)

    Solve for lambda > 0, then:
        u_right = (lambda - 1) * L
    """
    coeffs = [1.0, 0.0, -1.0, -(2.0 * traction / youngs_modulus)]
    roots = np.roots(coeffs)
    roots = roots[np.isclose(roots.imag, 0.0)].real
    roots = roots[roots > 0.0]
    if roots.size == 0:
        raise ValueError("No positive real stretch ratio found.")
    lam = roots[np.argmin(np.abs(roots - 1.0))]
    return (lam - 1.0) * length


def collect_curve(element_obj, traction_levels, youngs_modulus=1.0e03, poissons_ratio=0.0):
    mesh, _, _, right_nodes, L = build_bar_mesh()
    results = []

    for tr in traction_levels:
        sim = run_case(
            element_obj,
            traction_mag=float(tr),
            youngs_modulus=youngs_modulus,
            poissons_ratio=poissons_ratio,
        )
        ur = right_edge_mean_ux(sim, right_nodes)
        uref = finite_strain_bar_reference(float(tr), L, youngs_modulus)
        results.append((float(tr), ur, uref))

    return np.array(results, dtype=float)  # columns: traction, numerical ux, reference ux


def compare_cases():
    traction_levels = np.array([0.0, 100.0, 250.0, 500.0, 750.0], dtype=float)
    youngs_modulus = 1.0e03
    poissons_ratio = 0.0

    linear = collect_curve(felib.element.CPE4(), traction_levels, youngs_modulus, poissons_ratio)
    nonlinear = collect_curve(
        felib.element.CPE4NL(), traction_levels, youngs_modulus, poissons_ratio
    )

    return traction_levels, linear, nonlinear


def test_cpe4nl_is_closer_to_finite_strain_reference():
    traction_levels, linear, nonlinear = compare_cases()

    # Compare the largest load point, where the geometric nonlinearity matters most.
    lin_err = abs(linear[-1, 1] - linear[-1, 2])
    nl_err = abs(nonlinear[-1, 1] - nonlinear[-1, 2])

    print("Largest-load linear error   =", lin_err)
    print("Largest-load nonlinear error=", nl_err)

    assert nl_err < lin_err


def main():
    traction_levels, linear, nonlinear = compare_cases()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Load-displacement comparison
    axes[0].plot(linear[:, 0], linear[:, 1], "o-", label="CPE4 numerical")
    axes[0].plot(nonlinear[:, 0], nonlinear[:, 1], "s-", label="CPE4NL numerical")
    axes[0].plot(
        linear[:, 0], linear[:, 2], "k--", linewidth=2.0, label="Finite-strain bar reference"
    )
    axes[0].set_xlabel("Applied traction")
    axes[0].set_ylabel("Mean right-edge $u_x$")
    axes[0].set_title("Load-displacement comparison")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Absolute error comparison
    lin_err = np.abs(linear[:, 1] - linear[:, 2])
    nl_err = np.abs(nonlinear[:, 1] - nonlinear[:, 2])

    axes[1].plot(traction_levels, lin_err, "o-", label="CPE4 error")
    axes[1].plot(traction_levels, nl_err, "s-", label="CPE4NL error")
    axes[1].set_xlabel("Applied traction")
    axes[1].set_ylabel("Absolute error in $u_x$")
    axes[1].set_title("Error vs finite-strain reference")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.show()

    print("\nTraction   CPE4 ux   CPE4NL ux   Reference ux")
    for tr, (t1, u1, r1), (_, u2, r2) in zip(traction_levels, linear, nonlinear):
        print(f"{tr:8.1f}  {u1:8.4f}  {u2:9.4f}  {r1:11.4f}")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    plot_displacement_along_bar(traction=750.0)
