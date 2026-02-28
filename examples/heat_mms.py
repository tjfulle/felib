import argparse
import sys
from typing import Sequence

import numpy as np

import fem


def mms(esize: float = 0.05):
    class Everywhere(fem.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            return True

    class HeatSource(fem.collections.ScalarField):
        def __call__(self, x: Sequence[float], time: Sequence[float]) -> float:
            k = 12.0
            s = 24 * k * x[1] * (np.sin(12 * x[0] ** 2) + 24 * x[0] ** 2 * np.cos(12 * x[0] ** 2))
            return s

    nodes, elements = fem.meshing.uniform_plate(esize=esize)
    mesh_builder = fem.builder.MeshBuilder(nodes=nodes, elements=elements)
    mesh_builder.block(name="Block-1", region=Everywhere(), cell_type=fem.cell.Tri3)
    mesh_builder.elemset("All", region=Everywhere())
    mesh = mesh_builder.build()

    m = fem.material.HeatConduction(conductivity=12.0, specific_heat=1.0)
    builder = fem.builder.ModelBuilder(mesh, name="heat_mms")
    builder.assign_properties(block="Block-1", element=fem.element.DCP3(), material=m)
    step = builder.heat_transfer_step()
    step.source(elements="All", field=HeatSource())

    T = lambda x, y: np.cos(12 * x**2) * y
    p = np.asarray(nodes)
    x, y = p[:, 1], p[:, 2]
    mask = isclose(x, -1.0) | isclose(x, 1.0) | isclose(y, -1.0) | isclose(y, 1.0)
    for nid, x, y in p[mask]:
        step.temperature(nodes=int(nid), value=T(x, y))

    model = builder.build()
    model.solve()

    u = model.u[1]
    analytic = T(model.coords[:, 0], model.coords[:, 1])
    assert np.amax(np.abs(u - analytic)) < 0.03
    fem.plotting.tplot(model.coords, model.connect, u)
    fem.plotting.tplot3d(model.coords, model.connect, u)
    fem.plotting.tplot3d(
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
