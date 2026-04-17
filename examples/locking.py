import json

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import felib


def volume_locking_demo():
    class Everywhere(felib.collections.ElementSelector):
        def __call__(self, element: felib.collections.Element):
            return True

    mesh_data = json.load(open("QuarterCylinderQuad4.json"))
    mesh = felib.mesh.Mesh(nodes=mesh_data["nodes"], elements=mesh_data["elements"])
    mesh.block(name="Block-1", region=Everywhere(), cell_type=felib.element.Quad4)

    mesh.nodeset("Nodeset-200", nodes=mesh_data["nodesets"]["Nodeset-200"])
    mesh.nodeset("Nodeset-201", nodes=mesh_data["nodesets"]["Nodeset-201"])
    mesh.sideset("Surface-1", sides=mesh_data["sidesets"]["Surface-1"])

    mu, Nu = 1.0, 0.499
    E = 2.0 * mu * (1.0 + Nu)

    ul = cpe4(mesh, E, Nu, 1.0)
    ur = cpe4r(mesh, E, Nu, 1.0)
    ua = analytic_solution(mesh, E, Nu, 1.0)

    p, t = mesh.coords, mesh.connect
    _, ax = felib.plotting.mesh_plot_quad4(p + ua, t, label="Analytic", color="orange", lw=4)
    felib.plotting.mesh_plot_quad4(p + ul, t, label="Full Integration", color="b", ax=ax, ls="-.")
    felib.plotting.mesh_plot_quad4(
        p + ur, t, label="Reduced Integration", color="g", ax=ax, ls="--", lw=1.5
    )

    plt.legend(loc="best")
    plt.show()


def cpe4(mesh: felib.mesh.Mesh, E: float, Nu: float, pres: float) -> NDArray:
    model = felib.model.Model(mesh, name="volume_locking")
    material = felib.material.LinearElastic(density=2400.0, youngs_modulus=E, poissons_ratio=Nu)
    model.assign_properties(block="Block-1", element=felib.element.CPE4(), material=material)

    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step()

    step.boundary(nodes="Nodeset-200", dofs=[0], value=0.0)
    step.boundary(nodes="Nodeset-201", dofs=[1], value=0.0)
    step.pressure(sideset="Surface-1", magnitude=pres)

    simulation.run()
    u = simulation.ndata["u"]
    return u


def cpe4r(mesh: felib.mesh.Mesh, E: float, Nu: float, pres: float) -> NDArray:
    model = felib.model.Model(mesh, name="volume_locking")
    material = felib.material.LinearElastic(density=2400.0, youngs_modulus=E, poissons_ratio=Nu)
    model.assign_properties(block="Block-1", element=felib.element.CPE4R(), material=material)

    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step()

    step.boundary(nodes="Nodeset-200", dofs=[0], value=0.0)
    step.boundary(nodes="Nodeset-201", dofs=[1], value=0.0)
    step.pressure(sideset="Surface-1", magnitude=pres)

    simulation.run()
    u = simulation.ndata["u"]
    return u


def analytic_solution(mesh: felib.mesh.Mesh, E: float, Nu: float, pres) -> NDArray:
    a = mesh.coords[0, 1]
    b = mesh.coords[-1, 0]
    u = np.zeros_like(mesh.coords)
    for i, x in enumerate(mesh.coords):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        term1 = (1.0 + Nu) * a**2 * b**2 * pres / (E * (b**2 - a**2))
        term2 = 1.0 / r + (1.0 - 2.0 * Nu) * r / b**2
        ur = term1 * term2
        u[i, :] = ur * x[:] / r
    return u


if __name__ == "__main__":
    volume_locking_demo()
