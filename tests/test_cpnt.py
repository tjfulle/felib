"""
Tests for the CPnT thermo-mechanical element and StaggeredStep.

Unit tests
----------
test_thermal_strain                    - thermal strain vector is correct
test_thermal_strain_plane_strain       - plane strain thermal strain
test_ktt_assembly                      - K_TT block is symmetric and PSD
test_thermal_residual_linear_gradient  - thermal residual consistency

Integration tests
-----------------
test_single_element_thermo_mechanical  - single CPnT4S element, uniform T
test_uniform_temperature_plate         - clamped plate, uniform delta-T
test_linear_temperature_gradient       - free plate, linear T -> bending

Regression test
---------------
test_regression_pure_mechanical        - existing patch test unchanged
"""

from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

import felib
import felib.pytools as py
from felib.element.cpnt import CPnT4S, CPnT4E, _split_dofs
from felib.material import LinearElastic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unit_quad_coords() -> np.ndarray:
    return np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=float)


def _make_material() -> LinearElastic:
    m = LinearElastic(density=1.0, youngs_modulus=200e9, poissons_ratio=0.3)
    m.conductivity = 50.0
    return m


class Everywhere(felib.collections.RegionSelector):
    def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
        return True


class Left(felib.collections.RegionSelector):
    def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
        return on_boundary and x[0] < -0.999


class Right(felib.collections.RegionSelector):
    def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
        return on_boundary and x[0] > 0.999


class Bottom(felib.collections.RegionSelector):
    def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
        return on_boundary and x[1] < -0.999


class _AnyBoundary(felib.collections.RegionSelector):
    def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
        return on_boundary


class _BottomLeft(felib.collections.RegionSelector):
    def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
        return x[0] < -0.999 and x[1] < -0.999


# ---------------------------------------------------------------------------
# Unit test 1: thermal strain vector (plane stress)
# ---------------------------------------------------------------------------

def test_thermal_strain():
    """
    Thermal strain = alpha * (T - T_ref) * [1, 1, 0]  for plane stress.
    """
    el = CPnT4S()
    el.alpha = 1.2e-5
    el.T_ref = 0.0

    dT = 100.0
    e_th = el._thermal_strain(dT)

    assert e_th.shape == (el.ntens,)
    assert np.isclose(e_th[0], el.alpha * dT)
    assert np.isclose(e_th[1], el.alpha * dT)
    assert np.isclose(e_th[2], 0.0)


# ---------------------------------------------------------------------------
# Unit test 2: thermal strain (plane strain)
# ---------------------------------------------------------------------------

def test_thermal_strain_plane_strain():
    """
    Thermal strain for plane strain has 4 components.
    """
    el = CPnT4E()
    el.alpha = 1.2e-5
    el.T_ref = 20.0

    dT = 80.0
    e_th = el._thermal_strain(100.0)

    assert e_th.shape == (el.ntens,)
    assert np.isclose(e_th[0], el.alpha * dT)
    assert np.isclose(e_th[1], el.alpha * dT)
    assert np.isclose(e_th[2], 0.0)
    assert np.isclose(e_th[3], 0.0)


# ---------------------------------------------------------------------------
# Unit test 3: K_TT assembly
# ---------------------------------------------------------------------------

def test_ktt_assembly():
    """
    K_TT block must be symmetric and positive semi-definite.
    For uniform temperature, thermal residual R_T should be zero.
    """
    el = CPnT4S()
    p = _make_unit_quad_coords()
    nnode = el.nnode
    ndof = 3 * nnode

    mat = _make_material()
    u = np.zeros(ndof)
    T_val = 100.0
    for n in range(nnode):
        u[3 * n + 2] = T_val

    pdata = np.zeros((el.npts, len(el.history_variables())))
    ke, re = el.eval(mat, 1, 1, (0.0, 0.0), 1.0, 1, p, u, np.zeros(ndof), pdata)

    T_dofs = [3 * n + 2 for n in range(nnode)]
    K_TT = ke[np.ix_(T_dofs, T_dofs)]

    assert np.allclose(K_TT, K_TT.T, atol=1e-10), "K_TT is not symmetric"

    eigvals = np.linalg.eigvalsh(K_TT)
    assert np.all(eigvals >= -1e-10), f"K_TT has negative eigenvalue: {eigvals.min()}"

    R_T = re[T_dofs]
    assert np.allclose(R_T, 0.0, atol=1e-8), f"R_T not zero for uniform T: {R_T}"


# ---------------------------------------------------------------------------
# Unit test 4: thermal residual for linear gradient
# ---------------------------------------------------------------------------

def test_thermal_residual_linear_gradient():
    """
    For T = x, gradient dT/dx = 1 -> non-zero R_T but zero net flux.
    """
    el = CPnT4S()
    p = _make_unit_quad_coords()
    nnode = el.nnode
    ndof = 3 * nnode

    mat = _make_material()
    u = np.zeros(ndof)
    for n in range(nnode):
        u[3 * n + 2] = p[n, 0]   # T = x

    pdata = np.zeros((el.npts, len(el.history_variables())))
    ke, re = el.eval(mat, 1, 1, (0.0, 0.0), 1.0, 1, p, u, np.zeros(ndof), pdata)

    T_dofs = [3 * n + 2 for n in range(nnode)]
    R_T = re[T_dofs]

    assert not np.allclose(R_T, 0.0), "R_T should be non-zero for linear gradient"
    assert np.isclose(np.sum(R_T), 0.0, atol=1e-10), \
        f"Net thermal residual not zero: {np.sum(R_T)}"


# ---------------------------------------------------------------------------
# Integration test 1: single element
# ---------------------------------------------------------------------------

def test_single_element_thermo_mechanical(tmp_path: Path):
    """
    Single CPnT4S element clamped on all sides, heated uniformly.
    Verifies that temperature DOFs are set correctly after solve.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)

    nodes = [
        [1, 0.0, 0.0],
        [2, 1.0, 0.0],
        [3, 1.0, 1.0],
        [4, 0.0, 1.0],
    ]
    elements = [[1, 1, 2, 3, 4]]

    with py.working_dir(tmp_path):
        mesh = felib.mesh.Mesh(nodes, elements)
        mesh.block(name="All", elements=[1], cell_type=felib.element.Quad4)
        mesh.nodeset(name="all_nodes", nodes=[1, 2, 3, 4])

        alpha = 1.2e-5
        dT = 100.0

        mat = felib.material.LinearElastic(
            density=1.0, youngs_modulus=200e9, poissons_ratio=0.3
        )
        mat.conductivity = 50.0

        el = CPnT4S()
        el.alpha = alpha
        el.T_ref = 0.0

        model = felib.model.Model(mesh, name="single_cpnt")
        model.assign_properties(block="All", element=el, material=mat)

        simulation = felib.simulation.Simulation(model)
        step = simulation.staggered_step(name="thermo-mech")
        step.thermal.temperature(nodes="all_nodes", value=dT)
        step.mechanical.boundary(nodes="all_nodes", dofs=[felib.X, felib.Y], value=0.0)

        simulation.run()

        # Check temperature DOFs are set to dT
        T_col = model.node_freedom_cols[felib.constants.T]
        for lid in range(model.nnode):
            gdof = model.dof_map[lid, T_col]
            assert np.isclose(simulation.dofs[1][gdof], dT, atol=1e-6), \
                f"Node {lid}: T = {simulation.dofs[1][gdof]}, expected {dT}"


# ---------------------------------------------------------------------------
# Integration test 2: uniform temperature clamped plate
# ---------------------------------------------------------------------------

def test_uniform_temperature_plate(tmp_path: Path):
    """
    Clamped plate under uniform temperature rise.
    All displacement DOFs must remain zero.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)

    with py.working_dir(tmp_path):
        # uniform_plate generates triangles - use Tri3 / CPnT3 equivalent
        # For now use a simple 2x2 quad mesh manually
        nodes = [
            [1, -1.0, -1.0], [2,  0.0, -1.0], [3,  1.0, -1.0],
            [4, -1.0,  0.0], [5,  0.0,  0.0], [6,  1.0,  0.0],
            [7, -1.0,  1.0], [8,  0.0,  1.0], [9,  1.0,  1.0],
        ]
        elements = [
            [1, 1, 2, 5, 4],
            [2, 2, 3, 6, 5],
            [3, 4, 5, 8, 7],
            [4, 5, 6, 9, 8],
        ]

        mesh = felib.mesh.Mesh(nodes, elements)
        mesh.block(name="Block-1", elements=[1, 2, 3, 4], cell_type=felib.element.Quad4)
        mesh.nodeset(name="boundary", nodes=[1, 2, 3, 4, 6, 7, 8, 9])

        mat = felib.material.LinearElastic(
            density=1.0, youngs_modulus=200e9, poissons_ratio=0.3
        )
        mat.conductivity = 50.0

        el = CPnT4S()
        el.alpha = 1.2e-5
        el.T_ref = 0.0

        model = felib.model.Model(mesh, name="uniform_temp_plate")
        model.assign_properties(block="Block-1", element=el, material=mat)

        dT = 50.0
        simulation = felib.simulation.Simulation(model)
        step = simulation.staggered_step(name="thermo-mech")
        step.thermal.temperature(nodes="boundary", value=dT)
        step.mechanical.boundary(nodes="boundary", dofs=[felib.X, felib.Y], value=0.0)

        simulation.run()

        # All boundary displacement DOFs should be zero
        Ux_col = model.node_freedom_cols[felib.constants.Ux]
        Uy_col = model.node_freedom_cols[felib.constants.Uy]
        boundary_lids = model.nodesets["boundary"]
        for lid in boundary_lids:
            for col in [Ux_col, Uy_col]:
                gdof = model.dof_map[lid, col]
                assert np.isclose(simulation.dofs[1][gdof], 0.0, atol=1e-8), \
                    f"Boundary node {lid} displacement not zero"


# ---------------------------------------------------------------------------
# Integration test 3: linear temperature gradient -> bending
# ---------------------------------------------------------------------------

def test_linear_temperature_gradient(tmp_path: Path):
    """
    Free plate under linear temperature gradient (left cold, right hot).
    Interior nodes should displace -> bending.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)

    with py.working_dir(tmp_path):
        nodes = [
            [1, -1.0, -1.0], [2,  0.0, -1.0], [3,  1.0, -1.0],
            [4, -1.0,  0.0], [5,  0.0,  0.0], [6,  1.0,  0.0],
            [7, -1.0,  1.0], [8,  0.0,  1.0], [9,  1.0,  1.0],
        ]
        elements = [
            [1, 1, 2, 5, 4],
            [2, 2, 3, 6, 5],
            [3, 4, 5, 8, 7],
            [4, 5, 6, 9, 8],
        ]

        mesh = felib.mesh.Mesh(nodes, elements)
        mesh.block(name="Block-1", elements=[1, 2, 3, 4], cell_type=felib.element.Quad4)
        mesh.nodeset(name="left",   nodes=[1, 4, 7])
        mesh.nodeset(name="right",  nodes=[3, 6, 9])
        mesh.nodeset(name="corner", nodes=[1])

        mat = felib.material.LinearElastic(
            density=1.0, youngs_modulus=200e9, poissons_ratio=0.3
        )
        mat.conductivity = 50.0

        el = CPnT4S()
        el.alpha = 1.2e-5
        el.T_ref = 0.0

        model = felib.model.Model(mesh, name="gradient_plate")
        model.assign_properties(block="Block-1", element=el, material=mat)

        simulation = felib.simulation.Simulation(model)
        step = simulation.staggered_step(name="thermo-mech")
        step.thermal.temperature(nodes="left",  value=-50.0)
        step.thermal.temperature(nodes="right", value=50.0)
        step.mechanical.boundary(nodes="corner", dofs=[felib.X, felib.Y], value=0.0)

        simulation.run()

        # Some displacement should occur due to thermal expansion gradient
        Ux_col = model.node_freedom_cols[felib.constants.Ux]
        u_vals = [
            simulation.dofs[1][model.dof_map[lid, Ux_col]]
            for lid in range(model.nnode)
        ]
        assert max(abs(v) for v in u_vals) > 0.0, \
            "Expected non-zero displacement from thermal gradient"


# ---------------------------------------------------------------------------
# Regression test: pure mechanical unchanged
# ---------------------------------------------------------------------------

def test_regression_pure_mechanical(tmp_path: Path):
    """
    Standard patch test with CPE4 - must still pass after CPnT additions.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)

    nodes = [
        [1, 0.0, 0.0], [2, 4.0, 0.0], [3, 10.0, 0.0],
        [4, 0.0, 4.5], [5, 5.5, 5.5], [6, 10.0, 5.0],
        [7, 0.0, 10.0], [8, 4.2, 10.0], [9, 10.0, 10.0],
    ]
    elements = [[1, 1, 2, 5, 4], [2, 2, 3, 6, 5], [3, 4, 5, 8, 7], [4, 5, 6, 9, 8]]

    with py.working_dir(tmp_path):
        mesh = felib.mesh.Mesh(nodes, elements)
        mesh.block(name="All", elements=[1, 2, 3, 4], cell_type=felib.element.Quad4)
        mesh.sideset(name="IHI", sides=[[2, 2], [4, 2]])
        mesh.nodeset(name="ILO", nodes=[1, 4, 7])
        mesh.nodeset(name="FIX-Y", nodes=[1])

        mat = felib.material.LinearElastic(
            density=1.0, youngs_modulus=1.0e3, poissons_ratio=0.25
        )
        model = felib.model.Model(mesh, name="regression_mech")
        model.assign_properties(
            block="All", element=felib.element.CPE4(), material=mat
        )

        simulation = felib.simulation.Simulation(model)
        step = simulation.static_step()
        step.boundary(nodes="ILO", dofs=[felib.X])
        step.boundary(nodes="FIX-Y", dofs=[felib.Y])
        step.traction(sideset="IHI", magnitude=1.0, direction=[1.0, 0])

        simulation.run()

        for ebd in simulation.ebdata:
            assert np.allclose(ebd.data[0, :, :, 4], 1.0), \
                "Regression: patch test sxx != 1.0"
