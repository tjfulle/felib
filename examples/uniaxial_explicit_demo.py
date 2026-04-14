import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation
from matplotlib import colors, cm

import felib

X = felib.X
Y = felib.Y


def exercise(esize: float = 0.05):
    showVerificationPlots = True

    class Everywhere(felib.collections.ElementSelector):
        def __call__(self, element: felib.collections.Element):
            return True

    class Bottom(felib.collections.SideSelector):
        def __call__(self, side: felib.collections.Side):
            return side.x[1] < -0.999

    nodes, elements = felib.meshing.plate_with_hole(esize=esize)
    mesh = felib.mesh.Mesh(nodes=nodes, elements=elements)

    mesh.block(name="Block-1", region=Everywhere(), cell_type=felib.element.Tri3)
    mesh.nodeset("Point", region=lambda node: abs(node.x[0]) < 0.05 and node.x[1] > 0.999)
    mesh.nodeset("Top", region=lambda node: node.x[1] > 0.99)
    mesh.sideset("Bottom", region=Bottom())

    model = felib.model.Model(mesh, name="uniaxial_stress")

    material = felib.material.LinearElastic(
        density=2400.0,
        youngs_modulus=30.0e9,
        poissons_ratio=0.3,
    )
    model.assign_properties(
        block="Block-1",
        element=felib.element.CPS3(),
        material=material,
    )

    sim = felib.simulation.Simulation(model)
    step = sim.explicit_step(
        period=1.0e-3,
        damping=1.0,
        history_interval=1,
    )

    step.boundary(nodes="Point", dofs=[X, Y], value=0.0)
    step.boundary(nodes="Top", dofs=[Y], value=0.0)
    step.traction(sideset="Bottom", magnitude=1e8, direction=[0, -1])

    sim.run()

    u = sim.ndata["u"]
    cstep = sim.csteps[-1]

    times = np.array(cstep.time_history)
    u_hist = np.array(cstep.u_history)
    ke_hist = np.array(cstep.kinetic_energy_history)
    ie_hist = np.array(cstep.internal_energy_history)

    max_u_abs = np.max(np.abs(u))
    scale = 1.0 if max_u_abs == 0.0 else 0.25 / max_u_abs

    n_nodes = model.coords.shape[0]
    u_frames = u_hist.reshape((-1, n_nodes, 2))
    deformed_frames = model.coords[None, :, :] + scale * u_frames
    U_all = np.linalg.norm(u_frames, axis=2)

    vmin = np.min(U_all)
    vmax = np.max(U_all)

    xmin = np.min(deformed_frames[:, :, 0])
    xmax = np.max(deformed_frames[:, :, 0])
    ymin = np.min(deformed_frames[:, :, 1])
    ymax = np.max(deformed_frames[:, :, 1])

    triangles = model.connect

    print(f"Animation frames = {len(u_frames)}")

    if showVerificationPlots:
        tracked_node = np.argmax(np.linalg.norm(u, axis=1))
        uy_hist = u_hist[:, 2 * tracked_node + 1]
        total_energy = ke_hist + ie_hist

        plt.figure()
        plt.plot(times, uy_hist, lw=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Tracked node vertical displacement")
        plt.title("Tracked node displacement vs time")
        plt.grid(True)

        plt.figure()
        plt.plot(times, ke_hist, label="Kinetic energy")
        plt.plot(times, ie_hist, label="Internal energy")
        plt.plot(times, total_energy, label="Total energy", lw=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Energy")
        plt.title("Energy history")
        plt.grid(True)
        plt.legend()

    fig_anim, ax_anim = plt.subplots()

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    fig_anim.colorbar(sm, ax=ax_anim, label="Displacement magnitude")

    def update(frame: int):
        ax_anim.clear()

        coords = deformed_frames[frame]
        U_frame = U_all[frame]

        tri = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles)
        tpc = ax_anim.tripcolor(
            tri,
            U_frame,
            shading="gouraud",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax_anim.triplot(tri, color="k", linewidth=0.3)

        ax_anim.set_title(
            f"Uniaxial explicit deformation ({frame + 1}/{len(u_frames)}), "
            f"t = {times[frame]:.6e} s"
        )
        ax_anim.set_aspect("equal")
        ax_anim.set_xlim(xmin, xmax)
        ax_anim.set_ylim(ymin, ymax)

        return (tpc,)

    anim = FuncAnimation(
        fig_anim,
        update,
        frames=len(u_frames),
        interval=75,
        repeat=True,
    )

    plt.show()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-s", type=float, default=0.05, help="Element size [default: %(default)s]")
    args = p.parse_args()
    exercise(esize=args.s)
    return 0


if __name__ == "__main__":
    sys.exit(main())