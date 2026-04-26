import numpy as np

import felib

X = felib.X
Y = felib.Y


def uniaxial_tied():
    E = 210e9
    nu = 0.3
    traction_magnitude = 1e10  # positive x-direction traction on left face

    # --------------------
    # Tied two-block model (5x5 elements per block)
    # --------------------
    left_nodes, left_elems = felib.meshing.rectmesh((0.0, 1.0, 0.0, 1.0), nx=5, ny=5)
    right_nodes, right_elems = felib.meshing.rectmesh((1.0, 2.0, 0.0, 1.0), nx=5, ny=5)

    # Reindex right patch node/element ids to append after left
    max_node = len(left_nodes)
    max_elem = len(left_elems)
    right_nodes_shifted = [[nid + max_node, x, y] for nid, x, y in right_nodes]
    right_elems_shifted = [
        [eid + max_elem, n1 + max_node, n2 + max_node, n3 + max_node, n4 + max_node]
        for eid, n1, n2, n3, n4 in right_elems
    ]

    nodes_tied = left_nodes + right_nodes_shifted
    elems_tied = left_elems + right_elems_shifted

    mesh_tied = felib.mesh.Mesh(nodes=nodes_tied, elements=elems_tied)
    mesh_tied.block(name="Left", cell_type=felib.element.Quad4, elements=list(range(1, 5 * 5 + 1)))
    mesh_tied.block(
        name="Right", cell_type=felib.element.Quad4, elements=list(range(5 * 5 + 1, 2 * 5 * 5 + 1))
    )

    mesh_tied.sideset(name="LeftEdge", region=lambda side: np.isclose(side.x[0], 0.0))
    mesh_tied.nodeset(name="RightEdge", nodes=[n for n, x, y in nodes_tied if np.isclose(x, 2.0)])

    model_tied = felib.model.Model(mesh_tied, name="uniaxial_tied")
    m = felib.material.LinearElastic(density=0.0, youngs_modulus=E, poissons_ratio=nu)
    model_tied.assign_properties(block="Left", element=felib.element.CPE4(), material=m)
    model_tied.assign_properties(block="Right", element=felib.element.CPE4(), material=m)

    sim_tied = felib.simulation.Simulation(model_tied)
    step_tied = sim_tied.static_step()
    step_tied.boundary(nodes="RightEdge", dofs=[X, Y], value=0.0)
    step_tied.traction(sideset="LeftEdge", magnitude=traction_magnitude, direction=[1.0, 0.0])

    # Tie left/right at interface using equivalent homogeneous MPC eqns (node-to-node on x=1.0).
    left_interface = [(n, y) for n, x, y in nodes_tied if np.isclose(x, 1.0) and n <= max_node]
    right_interface = [(n, y) for n, x, y in nodes_tied if np.isclose(x, 1.0) and n > max_node]
    left_interface.sort(key=lambda pair: pair[1])
    right_interface.sort(key=lambda pair: pair[1])

    if len(left_interface) != len(right_interface):
        raise ValueError("Left/right interface node count mismatch for tied constraint")

    for (slave, _), (master, _) in zip(left_interface, right_interface):
        step_tied.equation(slave, X, 1.0, master, X, -1.0, 0.0)
        step_tied.equation(slave, Y, 1.0, master, Y, -1.0, 0.0)

    sim_tied.run()

    # --------------------------
    # Continuous single block ref (10x5 elements)
    # --------------------------
    nodes_cont, elems_cont = felib.meshing.rectmesh((0.0, 2.0, 0.0, 1.0), nx=10, ny=5)

    mesh_cont = felib.mesh.Mesh(nodes=nodes_cont, elements=elems_cont)
    mesh_cont.block(
        name="All", cell_type=felib.element.Quad4, elements=list(range(1, len(elems_cont) + 1))
    )
    mesh_cont.sideset(name="LeftEdge", region=lambda side: np.isclose(side.x[0], 0.0))
    mesh_cont.nodeset(
        name="RightEdge", nodes=[n for n, x, y in nodes_cont if np.isclose(x, 2.0)]
    )  # fixed boundary nodes at right edge

    model_cont = felib.model.Model(mesh_cont, name="uniaxial_cont")
    model_cont.assign_properties(block="All", element=felib.element.CPE4(), material=m)

    sim_cont = felib.simulation.Simulation(model_cont)
    step_cont = sim_cont.static_step()
    step_cont.boundary(nodes="RightEdge", dofs=[X, Y], value=0.0)
    step_cont.traction(sideset="LeftEdge", magnitude=traction_magnitude, direction=[1.0, 0.0])
    sim_cont.run()

    # --------------------------
    # Gather nodal and field data
    # --------------------------
    u_tied = sim_tied.ndata.gather("u")
    u_cont = sim_cont.ndata.gather("u")

    f_tied = sim_tied.csteps[-1].solution.force
    f_cont = sim_cont.csteps[-1].solution.force

    # Strain/stress from ebdata: [exx, eyy, exy, sxx, syy, sxy]
    def merge_elems(ebdata):
        arr = np.vstack([ebd.data[0, :, :] for ebd in ebdata])
        return arr[:, :3], arr[:, 3:6]

    strain_tied, stress_tied = merge_elems(sim_tied.ebdata)
    strain_cont, stress_cont = merge_elems(sim_cont.ebdata)

    # Compare at continuous nodes by mapping coordinate values
    def node_mapping(mesh_src, u_src):
        coords_src = np.asarray(mesh_src.coords)
        u_src = np.asarray(u_src)
        # aggregate tied duplicates at same coordinates (mean)
        unique = {}
        for i, c in enumerate(coords_src):
            key = (round(c[0], 8), round(c[1], 8))
            unique.setdefault(key, []).append(u_src[i])
        coords = []
        avg = []
        for k, vals in sorted(unique.items()):
            coords.append(np.array(k, dtype=float))
            avg.append(np.mean(vals, axis=0))
        return np.vstack(coords), np.vstack(avg)

    coords_tied_u, avg_tied_u = node_mapping(mesh_tied, u_tied)
    coords_cont_u, avg_cont_u = node_mapping(mesh_cont, u_cont)

    # Assume same coordinate set for comparison
    error_u = np.linalg.norm(avg_tied_u - avg_cont_u, axis=1)

    # Strain and stress error: for this coarse test compare averages per element
    # block centers are directly comparable by index (same number of elements = 2)
    strain_error = np.linalg.norm(strain_tied - strain_cont, axis=1)
    stress_error = np.linalg.norm(stress_tied - stress_cont, axis=1)

    print("=== Uniaxial tied vs continuous error report ===")
    print(
        f"Nodal displacement error (min/max/mean): {error_u.min():.3e} / {error_u.max():.3e} / {error_u.mean():.3e}"
    )
    print(
        f"Element strain error (min/max/mean): {strain_error.min():.3e} / {strain_error.max():.3e} / {strain_error.mean():.3e}"
    )
    print(
        f"Element stress error (min/max/mean): {stress_error.min():.3e} / {stress_error.max():.3e} / {stress_error.mean():.3e}"
    )
    print(
        f"Total force norm tied/cont: {np.linalg.norm(f_tied):.6e} / {np.linalg.norm(f_cont):.6e}"
    )

    print("Plot files saved:")
    print(f"  tied: {model_tied.name}.exo")
    print(f"  cont: {model_cont.name}.exo")

    return {
        "u_error": error_u,
        "strain_error": strain_error,
        "stress_error": stress_error,
        "force_tied": f_tied,
        "force_cont": f_cont,
    }


if __name__ == "__main__":
    results = uniaxial_tied()  # prints summary
