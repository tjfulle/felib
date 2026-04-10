from typing import Sequence

import numpy as np

import felib


def lame_constants(ri: float, ro: float, pi: float, po: float = 0.0) -> tuple[float, float]:
    a2 = ri**2
    b2 = ro**2
    A = (pi * a2 - po * b2) / (b2 - a2)
    B = a2 * b2 * (pi - po) / (b2 - a2)
    return A, B


def radial_displacement_exact(
    r: np.ndarray, ri: float, ro: float, pi: float, E: float, nu: float, po: float = 0.0
) -> np.ndarray:
    A, B = lame_constants(ri, ro, pi, po)
    return (1.0 + nu) / E * ((1.0 - 2.0 * nu) * A * r + B / r)


def hoop_stress_exact(
    r: np.ndarray, ri: float, ro: float, pi: float, po: float = 0.0
) -> np.ndarray:
    A, B = lame_constants(ri, ro, pi, po)
    return A + B / r**2


def test_axisymmetric_thick_cylinder():
    ri = 0.05
    ro = 0.10
    length = 0.20
    pressure = 100.0e6
    nr = 16
    nz = 8

    class Everywhere(felib.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            return True

    class InnerRadius(felib.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            return on_boundary and np.isclose(x[0], ri)

    class Bottom(felib.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            return on_boundary and np.isclose(x[1], 0.0)

    class Top(felib.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            return on_boundary and np.isclose(x[1], length)

    nodes, elements = felib.meshing.rectmesh((ri, ro, 0.0, length), nx=nr, ny=nz)
    mesh = felib.mesh.Mesh(nodes=nodes, elements=elements)
    mesh.block(name="Cylinder", region=Everywhere(), cell_type=felib.element.Quad4)
    mesh.sideset("Inner", region=InnerRadius())
    mesh.nodeset("Bottom", region=Bottom())
    mesh.nodeset("Top", region=Top())

    material = felib.material.LinearElastic(
        density=7800.0,
        youngs_modulus=210.0e9,
        poissons_ratio=0.3,
    )
    model = felib.model.Model(mesh, name="axisymmetric_thick_cylinder")
    model.assign_properties(block="Cylinder", element=felib.element.CAX4(), material=material)

    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step()
    step.boundary(nodes="Bottom", dofs=[felib.Y], value=0.0)
    step.boundary(nodes="Top", dofs=[felib.Y], value=0.0)
    step.pressure(sideset="Inner", magnitude=pressure)
    simulation.run()

    u = simulation.dofs[1].reshape((model.nnode, -1))
    ur = u[:, felib.X]
    radii = model.coords[:, felib.X]

    unique_r = np.unique(np.round(radii, 12))
    fe_ur = np.zeros_like(unique_r)
    for i, r in enumerate(unique_r):
        mask = np.isclose(radii, r)
        fe_ur[i] = np.mean(ur[mask])

    exact_ur = radial_displacement_exact(
        unique_r,
        ri,
        ro,
        pressure,
        material.youngs_modulus,
        material.poissons_ratio,
    )

    block = model.blocks[0]
    names = block.element_variable_names()
    hoop_idx = names.index("szz")
    elem_hoop = block.pdata[1, :, :, hoop_idx].mean(axis=1)

    nodal_hoop = np.zeros(model.nnode)
    nodal_count = np.zeros(model.nnode)
    for e, conn in enumerate(model.connect):
        nodal_hoop[conn] += elem_hoop[e]
        nodal_count[conn] += 1.0
    nodal_hoop /= np.maximum(nodal_count, 1.0)

    fe_hoop = np.zeros_like(unique_r)
    for i, r in enumerate(unique_r):
        mask = np.isclose(radii, r)
        fe_hoop[i] = np.mean(nodal_hoop[mask])

    exact_hoop = hoop_stress_exact(unique_r, ri, ro, pressure)

    ur_rel_error = np.max(np.abs(fe_ur - exact_ur)) / np.max(np.abs(exact_ur))
    hoop_rel_error = np.max(np.abs(fe_hoop - exact_hoop)) / np.max(np.abs(exact_hoop))

    assert ur_rel_error < 5e-2
    assert hoop_rel_error < 8e-2
