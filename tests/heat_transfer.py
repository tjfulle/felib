from pathlib import Path
from typing import Sequence

import numpy as np

import felib
import felib.pytools as py


def test_mms(tmp_path: Path):
    class Everywhere(felib.collections.ElementSelector):
        def __call__(self, element: felib.collections.Element):
            return True

    class HeatSource(felib.collections.ScalarField):
        def __call__(self, x: Sequence[float], time: Sequence[float]) -> float:
            k = 12.0
            s = 24 * k * x[1] * (np.sin(12 * x[0] ** 2) + 24 * x[0] ** 2 * np.cos(12 * x[0] ** 2))
            return s

    tmp_path.mkdir(parents=True, exist_ok=True)
    with py.working_dir(tmp_path):
        nodes, elements = felib.meshing.uniform_plate(esize=0.05)
        mesh = felib.mesh.Mesh(nodes=nodes, elements=elements)
        mesh.block(name="Block-1", region=Everywhere(), cell_type=felib.element.Tri3)
        mesh.elemset("All", region=Everywhere())

        m = felib.material.HeatConduction(conductivity=12.0, specific_heat=1.0)
        model = felib.model.Model(mesh, name="heat_mms")
        model.assign_properties(block="Block-1", element=felib.element.DCP3(), material=m)

        simulation = felib.simulation.Simulation(model)
        step = simulation.heat_transfer_step()
        step.source(elements="All", field=HeatSource())

        T = lambda x, y: np.cos(12 * x**2) * y
        p = np.asarray(nodes)
        x, y = p[:, 1], p[:, 2]
        mask = isclose(x, -1.0) | isclose(x, 1.0) | isclose(y, -1.0) | isclose(y, 1.0)
        for nid, x, y in p[mask]:
            step.temperature(nodes=int(nid), value=T(x, y))

        simulation.run()

        # Choose elements on the interior to neglect edge effects
        ix = np.where((np.abs(model.coords[:, 0]) < 0.8) & (np.abs(model.coords[:, 1]) < 0.8))[0]
        dofs = simulation.ndata.gather_dofs()
        u = dofs[ix]
        analytic = T(model.coords[ix, 0], model.coords[ix, 1])
        assert np.amax(np.abs(u - analytic)) < 1.5e-3


def test_heat1(tmp_path: Path):
    """
    Solve the 2D heat equation over a square domain

    • Bounds: x ∈ [-1, 1], y ∈ [-1, 1] with hole of radius .3 at its center.
    • Spatially varying heat source: 1000 / √(x^2 + y^2)
    • Fixed temperature on left edge: 200˚
    • Fixed temperature on right edge: 50˚
    • Heat flux along bottom edge: 2000
    • Convection along top edge: far field temperature 25˚ with convection coefficient 250

    """

    class Everywhere(felib.collections.ElementSelector):
        def __call__(self, element: felib.collections.Element):
            return True

    class Top(felib.collections.SideSelector):
        def __call__(self, side: felib.collections.Side) -> bool:
            return side.x[1] > 0.999

    class Bottom(felib.collections.SideSelector):
        def __call__(self, side: felib.collections.Side) -> bool:
            return side.x[1] < -0.999

    tmp_path.mkdir(parents=True, exist_ok=True)
    with py.working_dir(tmp_path):
        nodes, elements = felib.meshing.uniform_plate(esize=0.05)
        mesh = felib.mesh.Mesh(nodes=nodes, elements=elements)
        mesh.block(name="Block-1", region=Everywhere(), cell_type=felib.element.Tri3)
        mesh.sideset("Top", region=Top())
        mesh.sideset("Bottom", region=Bottom())
        mesh.elemset("All", region=Everywhere())

        m = felib.material.HeatConduction(conductivity=12.0, specific_heat=1.0)
        model = felib.model.Model(mesh, name="heat1")
        model.assign_properties(block="Block-1", element=felib.element.DCP3(), material=m)

        simulation = felib.simulation.Simulation(model)
        step = simulation.heat_transfer_step()
        step.film(sideset="Top", h=250.0, ambient_temp=25.0)
        step.dflux(sideset="Bottom", magnitude=2000.0, direction=[0.0, 1.0])
        simulation.run()
        u = simulation.ndata.gather_dofs()
        thi = u[np.where(np.abs(mesh.coords[:, 1] - 1.0) < 1e-6)[0]]
        assert np.allclose(thi, 33)
        tlo = u[np.where(np.abs(mesh.coords[:, 1] + 1.0) < 1e-6)[0]]
        assert np.allclose(tlo, 366.33333)


def isclose(a, b):
    return np.isclose(a, b, atol=1e-12)
