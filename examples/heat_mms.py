import argparse
import sys
from typing import Sequence

import numpy as np

import felib


def mms(esize: float = 0.05):
    class Everywhere(felib.collections.ElementSelector):
        def __call__(self, el: felib.collections.Element):
            return True

    class HeatSource(felib.collections.ScalarField):
        def __call__(self, x: Sequence[float], time: Sequence[float]) -> float:
            k = 12.0
            s = 24 * k * x[1] * (np.sin(12 * x[0] ** 2) + 24 * x[0] ** 2 * np.cos(12 * x[0] ** 2))
            return s

    nodes, elements = felib.meshing.uniform_plate(esize=esize)
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

    u = simulation.ndata["T"]
    analytic = T(model.coords[:, 0], model.coords[:, 1])
    assert np.amax(np.abs(u - analytic)) < 0.03
    felib.plotting.tplot(model.coords, model.connect, u)
    felib.plotting.tplot3d(model.coords, model.connect, u)
    felib.plotting.tplot3d(
        model.coords,
        model.connect,
        u - analytic,
        label="$T - T_{ana}$",
        title="Error in FEA solution",
    )


def isclose(a, b):
    return np.isclose(a, b, atol=1e-12)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-s", type=float, default=0.05, help="Element size [default: %(default)s]")
    args = p.parse_args()
    mms(esize=args.s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
