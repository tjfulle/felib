import numpy as np

import felib

X = felib.X
Y = felib.Y


def run_example():
    E = 1
    nu = 0.3
    q = -1.0  # point load magnitude (negative Y)

    # --- Tied interface case (2 blocks, MPC-equivalent via equation ties) ---
    nodes_tied = [
        [1, 0.0, 0.0],
        [2, 1.0, 0.0],
        [3, 1.0, 1.0],
        [4, 0.0, 1.0],
        [5, 1.0, 0.0],
        [6, 2.0, 0.0],
        [7, 2.0, 1.0],
        [8, 1.0, 1.0],
    ]
    elems_tied = [[1, 1, 2, 3, 4], [2, 5, 6, 7, 8]]

    mesh_tied = felib.mesh.Mesh(nodes=nodes_tied, elements=elems_tied)
    mesh_tied.block(name="Left", cell_type=felib.element.Quad4, elements=[1])
    mesh_tied.block(name="Right", cell_type=felib.element.Quad4, elements=[2])

    model_tied = felib.model.Model(mesh_tied, name="tied_mpc")
    mat = felib.material.LinearElastic(density=0.0, youngs_modulus=E, poissons_ratio=nu)
    model_tied.assign_properties(block="Left", element=felib.element.CPE4(), material=mat)
    model_tied.assign_properties(block="Right", element=felib.element.CPE4(), material=mat)

    sim_tied = felib.simulation.Simulation(model_tied)
    step_tied = sim_tied.static_step()
    step_tied.boundary(nodes=[1, 4], dofs=[X, Y], value=0.0)
    step_tied.point_load(nodes=[6, 7], dofs=Y, value=q)

    # Homogeneous tie as equation constraints (mini MPC implementation)
    for slave, master in [(2, 5), (3, 8)]:
        step_tied.equation(slave, X, 1.0, master, X, -1.0, 0.0)
        step_tied.equation(slave, Y, 1.0, master, Y, -1.0, 0.0)

    sim_tied.run()

    # --- Continuous single-block case (reference solution) ---
    nodes_cont = [
        [1, 0.0, 0.0],
        [2, 1.0, 0.0],
        [3, 1.0, 1.0],
        [4, 0.0, 1.0],
        [5, 2.0, 0.0],
        [6, 2.0, 1.0],
    ]
    elems_cont = [[1, 1, 2, 3, 4], [2, 2, 5, 6, 3]]

    mesh_cont = felib.mesh.Mesh(nodes=nodes_cont, elements=elems_cont)
    mesh_cont.block(name="All", cell_type=felib.element.Quad4, elements=[1, 2])

    model_cont = felib.model.Model(mesh_cont, name="cont_mpc")
    model_cont.assign_properties(block="All", element=felib.element.CPE4(), material=mat)

    sim_cont = felib.simulation.Simulation(model_cont)
    step_cont = sim_cont.static_step()
    step_cont.boundary(nodes=[1, 4], dofs=[X, Y], value=0.0)
    step_cont.point_load(nodes=[5, 6], dofs=Y, value=q)

    sim_cont.run()

    # --- Data collection ---
    u_tied = sim_tied.ndata.gather("u")
    u_cont = sim_cont.ndata.gather("u")

    # Compute force vectors from last step solutions
    f_tied = sim_tied.csteps[-1].solution.force
    f_cont = sim_cont.csteps[-1].solution.force

    # Element strains (exx, eyy, exy) for each block
    strains_tied = [ebd.data[0, :, 0:3] for ebd in sim_tied.ebdata]
    strains_cont = [ebd.data[0, :, 0:3] for ebd in sim_cont.ebdata]

    # Map tied-interface node displacements to continuous counterpart
    tied_interface = [(2, 2), (3, 3), (5, 2), (8, 3)]
    tied_coords = []
    tied_disp = []
    cont_disp = []
    for tied_idx, cont_idx in tied_interface:
        tied_coords.append(mesh_tied.coords[tied_idx - 1].tolist())
        tied_disp.append(u_tied[tied_idx - 1].tolist())
        cont_disp.append(u_cont[cont_idx - 1].tolist())

    tied_disp = np.array(tied_disp)
    cont_disp = np.array(cont_disp)
    disp_diff = tied_disp - cont_disp
    error_norm = np.linalg.norm(disp_diff, axis=1)

    print("=== Tied vs Continuous MPC example ===")
    print(f"Tied model exodus: {model_tied.name}.exo")
    print(f"Continuous model exodus: {model_cont.name}.exo")
    print(f"Total nodes (tied): {u_tied.shape[0]}, (cont): {u_cont.shape[0]}")

    print("--- Displacement (tied) at interface points ---")
    for i, (coord, td, cd, diff, err) in enumerate(
        zip(tied_coords, tied_disp, cont_disp, disp_diff, error_norm)
    ):
        print(
            f"  Pt {i}: coord={coord}, tied={td}, cont={cd}, diff={diff}, err={err:.3e}"
        )

    print("--- Max / mean interface disp error ---")
    print(f"  max error = {np.max(error_norm):.3e}")
    print(f"  mean error = {np.mean(error_norm):.3e}")

    print("--- Force norm comparison ---")
    print(f"  tied force norm = {np.linalg.norm(f_tied):.6e}")
    print(f"  cont force norm = {np.linalg.norm(f_cont):.6e}")

    print("--- Element strain summary ---")
    print(f"  tied strain blocks: {[s.shape for s in strains_tied]}")
    print(f"  cont strain blocks: {[s.shape for s in strains_cont]}")
    print("  (strain variables are exx, eyy, exy per element centroid)")

    return {
        "tied": {
            "u": u_tied,
            "force": f_tied,
            "strains": strains_tied,
        },
        "cont": {
            "u": u_cont,
            "force": f_cont,
            "strains": strains_cont,
        },
        "interface": {
            "coords": np.array(tied_coords),
            "tied": tied_disp,
            "cont": cont_disp,
            "diff": disp_diff,
            "error_norm": error_norm,
        },
    }


if __name__ == "__main__":
    run_example()
