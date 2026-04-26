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


def test_linear_elastic_plane_strain_tangent_matches_closed_form():
    mat = LinearElastic(youngs_modulus=1000.0, poissons_ratio=0.3)
    e = np.array([0.02, -0.01, 0.0, 0.03])

    D, s = mat.eval(
        np.zeros(0),
        e,
        np.zeros_like(e),
        time=(0.0, 1.0),
        dtime=1.0,
        temp=0.0,
        dtemp=0.0,
        ndir=3,
        nshr=1,
        eleno=1,
        step=1,
        increment=1,
    )

    E = mat.youngs_modulus
    nu = mat.poissons_ratio
    factor = E / ((1 + nu) * (1 - 2 * nu))
    expected_D = factor * np.array(
        [
            [1 - nu, nu, nu, 0.0],
            [nu, 1 - nu, nu, 0.0],
            [nu, nu, 1 - nu, 0.0],
            [0.0, 0.0, 0.0, (1 - 2 * nu) / 2],
        ]
    )

    assert np.allclose(D, expected_D)
    assert np.allclose(s, expected_D @ e)


def test_linear_elastic_hybrid_split_reconstructs_plane_strain_response():
    mat = LinearElastic(youngs_modulus=1000.0, poissons_ratio=0.3)
    e = np.array([0.02, -0.01, 0.0, 0.03])

    D, s = mat.eval(
        np.zeros(0),
        e,
        np.zeros_like(e),
        time=(0.0, 1.0),
        dtime=1.0,
        temp=0.0,
        dtemp=0.0,
        ndir=3,
        nshr=1,
        eleno=1,
        step=1,
        increment=1,
    )
    Ddev, sdev, Kbulk = mat.eval_hybrid(
        np.zeros(0),
        e,
        np.zeros_like(e),
        time=(0.0, 1.0),
        dtime=1.0,
        temp=0.0,
        dtemp=0.0,
        ndir=3,
        nshr=1,
        eleno=1,
        step=1,
        increment=1,
    )

    volumetric_direction = np.array([1.0, 1.0, 1.0, 0.0])
    volumetric_projector = np.outer(volumetric_direction, volumetric_direction)
    reconstructed_D = Ddev + Kbulk * volumetric_projector
    reconstructed_s = sdev + Kbulk * np.sum(e[:3]) * volumetric_direction

    assert np.allclose(reconstructed_D, D)
    assert np.allclose(reconstructed_s, s)


def test_cpe4h_condensed_pressure_matches_bulk_response():
    el = CPE4H()
    mat = LinearElastic(youngs_modulus=1000.0, poissons_ratio=0.3)
    p = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
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

    E = mat.youngs_modulus
    nu = mat.poissons_ratio
    Kbulk = E / (3 * (1 - 2 * nu))
    ev = 0.01
    expected_pressure = -Kbulk * ev

    expected_strain = np.array([0.02, -0.01, 0.0, 0.0])
    factor = E / ((1 + nu) * (1 - 2 * nu))
    expected_stress = factor * np.array(
        [
            (1 - nu) * expected_strain[0] + nu * expected_strain[1] + nu * expected_strain[2],
            nu * expected_strain[0] + (1 - nu) * expected_strain[1] + nu * expected_strain[2],
            nu * expected_strain[0] + nu * expected_strain[1] + (1 - nu) * expected_strain[2],
            ((1 - 2 * nu) / 2) * expected_strain[3],
        ]
    )

    assert ke.shape == (8, 8)
    assert re.shape == (8,)
    assert np.allclose(pdata[:, :4], expected_strain)
    assert np.allclose(pdata[:, 4:8], expected_stress)
    assert np.allclose(pdata[:, 8], expected_pressure)
    assert np.allclose(pdata[:, 9], ev)


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
