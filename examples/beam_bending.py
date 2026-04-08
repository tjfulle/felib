import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import felib

X = felib.X
Y = felib.Y

mu = 10000.0
nu = 0.0
E = 2.0 * mu * (1.0 + nu)


def beam_bending() -> None:

    q4 = beam_bending_quad4()
    q4i = beam_bending_quad4i()
    q8 = beam_bending_quad8()

    u4 = q4.ndata["u"]
    u4i = q4i.ndata["u"]
    u8 = q8.ndata["u"]

    scale = 0.25 / np.max(np.abs(u4))
    ua = analytic_solution(q4.model.mesh)

    x4 = q4.model.coords
    c4 = q4.model.connect
    _, ax = felib.plotting.mesh_plot(
        felib.element.Quad4(), x4 + scale * ua, c4, label="Analytic solution", color="orange", lw=4
    )
#    felib.plotting.mesh_plot(
#        felib.element.Quad4(), x4 + scale * u4, c4, label="FE CPS4 Solution", ax=ax
#    )
    x4i = q4i.model.coords
    c4i = q4i.model.connect
    felib.plotting.mesh_plot(
        felib.element.Quad4(), x4i + scale * u4i, c4i, label="FE CPS4I Solution", ax=ax, ls="-.", color="b"
    )

#    x8 = q8.model.coords
#    c8 = q8.model.connect
#    felib.plotting.mesh_plot(
#        felib.element.Quad8(), x8 + scale * u8, c8, label="FE CPS8 Solution", ax=ax, color="b"
#    )

    ax.set_aspect("equal")
    plt.legend(loc="best")
    plt.show()


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
    step = simulation.static_step()
    step.boundary(nodes="ihi", dofs=[X, Y], value=0.0)
    step.traction(sideset="ilo", magnitude=1, direction=[0, -1])
    simulation.run()
    return simulation

def beam_bending_quad4i() -> felib.simulation.Simulation:
    nodes, elems = felib.meshing.rectmesh((0, 10.0, 0, 0.3), nx=10, ny=3)
    mesh = felib.mesh.Mesh(nodes=nodes, elements=elems)
    mesh.nodeset(name="ihi", region=lambda node: node.on_boundary and node.x[0] > 9.991)
    mesh.sideset(name="ilo", region=lambda side: side.x[0] < 0.001)
    mesh.block(name="Block-1", cell_type=felib.element.Quad4, region=lambda el: True)
    m = felib.material.LinearElastic(density=2400.0, youngs_modulus=E, poissons_ratio=nu)
    model = felib.model.Model(mesh, name="shear_locking")
    model.assign_properties(block="Block-1", element=felib.element.CPS4I(), material=m)
    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step()
    step.boundary(nodes="ihi", dofs=[X, Y], value=0.0)
    step.traction(sideset="ilo", magnitude=1, direction=[0, -1])
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
    step = simulation.static_step()
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
