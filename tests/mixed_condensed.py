import numpy as np

from felib.element.cnd import CPE4
from felib.element.cnd import CPE4H
from felib.material import LinearElastic


class CPE4LocalP(CPE4):
    uses_local_pressure = True
    npressure = 4

    def history_variables(self) -> list[str]:
        names = super().history_variables()
        names.extend(["p1", "p2", "p3", "p4", "ev"])
        return names

    def bmatrix_vol(self, p, xi):
        dNdx = self.shape_gradient(p, xi)
        Bv = np.zeros((1, 8))
        Bv[0, 0::2] = dNdx[0]
        Bv[0, 1::2] = dNdx[1]
        return Bv

    def pressure_shape(self, xi):
        return np.atleast_2d(self.shape(xi))


def test_cpe4h_condensed_eval():
    el = CPE4H()
    mat = LinearElastic(youngs_modulus=1000.0, poissons_ratio=0.499)
    p = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    u = np.zeros(8)
    du = np.zeros(8)
    pdata = np.zeros((el.npts, len(el.history_variables())))

    ke, re = el.eval(
        mat,
        step=1,
        increment=1,
        time=(0.0, 1.0),
        dt=1.0,
        eleno=1,
        p=p,
        u=u,
        du=du,
        pdata=pdata,
    )

    assert ke.shape == (8, 8)
    assert re.shape == (8,)
    assert np.isfinite(ke).all()
    assert np.isfinite(re).all()
    assert np.isfinite(pdata).all()


def test_local_nodal_pressure_condensed_eval():
    el = CPE4LocalP()
    mat = LinearElastic(youngs_modulus=1000.0, poissons_ratio=0.3)
    p = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    u = np.array([0.0, 0.0, 0.02, 0.0, 0.02, -0.01, 0.0, -0.01])
    du = np.zeros(8)
    pdata = np.zeros((el.npts, len(el.history_variables())))

    ke, re = el.eval(
        mat,
        step=1,
        increment=1,
        time=(0.0, 1.0),
        dt=1.0,
        eleno=1,
        p=p,
        u=u,
        du=du,
        pdata=pdata,
    )

    assert ke.shape == (8, 8)
    assert re.shape == (8,)
    assert np.isfinite(ke).all()
    assert np.isfinite(re).all()
    assert np.isfinite(pdata).all()
