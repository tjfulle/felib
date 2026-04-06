import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from matplotlib.animation import FuncAnimation

import felib

X = felib.X
Y = felib.Y

mu = 10000.0
nu = 0.0
E = 2.0 * mu * (1.0 + nu)

def beam_bending() -> None:
    q4 = beam_bending_quad4()
    u4 = q4.ndata["u"]

    coords = q4.model.coords
    tip_node = np.argmin(np.linalg.norm(coords - np.array([0.0, 0.15]), axis=1))

    cstep = q4.csteps[-1]
    times = np.array(cstep.time_history)
    u_hist = np.array(cstep.u_history)
    uy_tip = u_hist[:, 2 * tip_node + 1]
    ke_hist = np.array(cstep.kinetic_energy_history)
    ie_hist = np.array(cstep.internal_energy_history)

    print("tracked node index =", tip_node)
    print("num history frames =", len(times))
    print("max |u| =", np.max(np.linalg.norm(u4, axis=1)))

    scale = 0.25 / np.max(np.abs(u4))
    x4 = q4.model.coords
    c4 = q4.model.connect

    _, ax = felib.plotting.mesh_plot(
        felib.element.Quad4(),
        x4 + scale * u4,
        c4,
        label="FE CPS4 Explicit Solution",
    )
    ax.set_aspect("equal")
    plt.legend(loc="best")
    plt.show()

    # plt.figure()
    # plt.plot(times, uy_tip)
    # plt.xlabel("Time")
    # plt.ylabel("Vertical displacement")
    # plt.title("Beam tip displacement vs time")
    # plt.show()

    plt.figure()
    plt.plot(times, ke_hist)
    plt.xlabel("Time")
    plt.ylabel("Kinetic energy")
    plt.title("Kinetic energy vs time")
    plt.show()

    plt.figure()
    plt.plot(times, ie_hist)
    plt.xlabel("Time")
    plt.ylabel("Internal energy")
    plt.title("Internal energy vs time")
    plt.show()

    plt.figure()
    plt.plot(times, ke_hist + ie_hist)
    plt.xlabel("Time")
    plt.ylabel("Total energy")
    plt.title("Total energy vs time")
    plt.show()

    u_frames = np.array(cstep.u_history)

    fig_anim, ax_anim = plt.subplots()

    def update(frame: int):
        ax_anim.clear()
        u_frame = u_frames[frame].reshape((-1, 2))
        felib.plotting.mesh_plot(
            felib.element.Quad4(),
            x4 + scale * u_frame,
            c4,
            ax=ax_anim,
            label=f"Frame {frame + 1}/{len(u_frames)}",
        )
        ax_anim.set_aspect("equal")
        ax_anim.set_title("Beam deformation history")
        ax_anim.legend(loc="best")

    anim = FuncAnimation(fig_anim, update, frames=len(u_frames), interval=100, repeat=True)
    _ = anim
    plt.show()

# def beam_bending() -> None:

#     q4 = beam_bending_quad4()
#     q8 = beam_bending_quad8()

#     u4 = q4.ndata["u"]
#     u8 = q8.ndata["u"]

#     scale = 0.25 / np.max(np.abs(u4))
#     ua = analytic_solution(q4.model.mesh)

#     x4 = q4.model.coords
#     c4 = q4.model.connect
#     _, ax = felib.plotting.mesh_plot(
#         felib.element.Quad4(), x4 + scale * ua, c4, label="Analytic solution", color="r"
#     )
#     felib.plotting.mesh_plot(
#         felib.element.Quad4(), x4 + scale * u4, c4, label="FE CPS4 Solution", ax=ax
#     )

#     x8 = q8.model.coords
#     c8 = q8.model.connect
#     felib.plotting.mesh_plot(
#         felib.element.Quad8(), x8 + scale * u8, c8, label="FE CPS8 Solution", ax=ax, color="b"
#     )

#     ax.set_aspect("equal")
#     plt.legend(loc="best")
#     plt.show()


def beam_bending_quad4() -> felib.simulation.Simulation:
    nodes, elems = felib.meshing.rectmesh((0, 10.0, 0, 0.3), nx=10, ny=3)
    mesh = felib.mesh.Mesh(nodes=nodes, elements=elems)
    mesh.nodeset(name="ihi", region=lambda node: node.on_boundary and node.x[0] > 9.991)
    mesh.sideset(name="ilo", region=lambda side: side.x[0] < 0.001)
    mesh.block(name="Block-1", cell_type=felib.element.Quad4, region=lambda el: True)
    m = felib.material.LinearElastic(density=2400.0, youngs_modulus=E, poissons_ratio=nu)
    model = felib.model.Model(mesh, name="shear_locking")
    model.assign_properties(block="Block-1", element=felib.element.CPS4(), material=m)
    simulation = felib.simulation.Simulation(model)
    step = simulation.explicit_step(
        period=1.0e-2,
        dt=1.0e-6,
        damping=1.0,
        history_interval=100,
    )
    step.boundary(nodes="ihi", dofs=[X, Y], value=0.0)
    step.traction(sideset="ilo", magnitude=1e6, direction=[0, -1])
    simulation.run()
    return simulation


def beam_bending_quad8() -> felib.simulation.Simulation:
    nodes, elems = felib.meshing.rectmesh_quad8((0, 10.0, 0, 0.3), nx=10, ny=3)
    mesh = felib.mesh.Mesh(nodes=nodes, elements=elems)
    mesh.nodeset(name="ihi", region=lambda node: node.on_boundary and node.x[0] > 9.991)
    mesh.sideset(name="ilo", region=lambda side: side.x[0] < 0.001)
    mesh.block(name="Block-1", cell_type=felib.element.Quad8, region=lambda el: True)
    m = felib.material.LinearElastic(density=2400.0, youngs_modulus=E, poissons_ratio=nu)
    model = felib.model.Model(mesh, name="shear_locking")
    model.assign_properties(block="Block-1", element=felib.element.CPS8(), material=m)
    simulation = felib.simulation.Simulation(model)
    step = simulation.explicit_step(period=1.0e-4)
    step.boundary(nodes="ihi", dofs=[X, Y], value=0.0)
    step.traction(sideset="ilo", magnitude=1, direction=[0, -1])
    simulation.run()
    return simulation


def analytic_solution(mesh: felib.mesh.Mesh) -> NDArray:
    a = 0.15
    b = 1.0
    P = 2 * a * b
    L = 10.0
    d = -P * L**3 / (2 * E * a**3 * b)
    II = b * (2.0 * a) ** 3 / 12.0
    u = np.zeros_like(mesh.coords)
    for i, x in enumerate(mesh.coords):
        x1, x2 = x
        x2 -= a
        u[i, 0] = (
            P / (2.0 * E * II) * x1**2 * x2
            + nu * P / (6 * E * II) * x2**3
            - P / (6 * II * mu) * x2**3
            - (P * L**2 / (2 * E * II) - P * a**2 / (2 * II * mu)) * x2
        )
        u[i, 1] = (
            -nu * P / (2 * E * II) * x1 * x2**2
            - P / (6 * E * II) * x1**3
            + P * L**2 / (2 * E * II) * x1
            - P * L**3 / (3 * E * II)
        )
    return u


if __name__ == "__main__":
    sys.exit(beam_bending())
