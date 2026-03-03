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


def beam_bending():

    nodes, elems = felib.meshing.rectmesh((0, 10.0, 0, 0.3), nx=10, ny=3)
    mesh = felib.mesh.Mesh(nodes=nodes, elements=elems)
    mesh.nodeset(name="nodeset-1", region=lambda x, on_boundary: on_boundary and x[0] > 9.991)
    mesh.sideset(name="sideset-1", region=lambda x, on_boundary: on_boundary and x[0] < 0.001)
    mesh.block(name="Block-1", cell_type=felib.cell.Quad4, region=lambda x, on_boundary: True)

    #    mesh = make_mesh()

    m = felib.material.LinearElastic(density=2400.0, youngs_modulus=E, poissons_ratio=nu)
    model = felib.model.Model(mesh, name="shear_locking")
    model.assign_properties(block="Block-1", element=felib.element.CPS4(), material=m)

    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step()
    step.boundary(nodes="nodeset-1", dofs=[X, Y], value=0.0)
    #    step.boundary(nodes="nodeset-2", dofs=[X], value=0.0)

    step.traction(sideset="sideset-1", magnitude=1, direction=[0, -1])

    simulation.run()

    u = simulation.dofs[1].reshape(model.nnode, -1)

    scale = 0.25 / np.max(np.abs(u))
    _, ax = felib.plotting.mesh_plot_quad4(
        model.coords + scale * u, model.connect, label="FE CPS4 Solution"
    )

    ua = analytic_solution(mesh)
    felib.plotting.mesh_plot_quad4(
        model.coords + scale * ua, model.connect, ax=ax, label="Analytic solution", color="r"
    )

    ax.set_aspect("equal")
    plt.legend(loc="best")
    plt.show()


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
