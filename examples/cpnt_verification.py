"""
cpnt_verification.py
====================
Verification examples for the CPnT sequential thermo-mechanical element.

Three cases from Issue #6:
  1. Uniform temperature plate with clamped edges  -> thermal stresses
  2. Linear temperature gradient                   -> bending of free plate
  3. Multi-material block                          -> stress distribution

Run:
    python examples/cpnt_verification.py --case 1   # uniform temperature
    python examples/cpnt_verification.py --case 2   # linear gradient
    python examples/cpnt_verification.py --case 3   # multi-material
    python examples/cpnt_verification.py            # all three
"""

import argparse
import sys
from typing import Sequence

import numpy as np

np.set_printoptions(precision=4)

import felib
from felib.element.cpnt import CPnT4S

X = felib.X
Y = felib.Y


# ---------------------------------------------------------------------------
# Region selectors
# ---------------------------------------------------------------------------

class Left(felib.collections.RegionSelector):
    def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
        return on_boundary and x[0] < -0.999


class Right(felib.collections.RegionSelector):
    def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
        return on_boundary and x[0] > 0.999


# ---------------------------------------------------------------------------
# Helper: quad connectivity -> triangle connectivity for plotting
# ---------------------------------------------------------------------------

def _quads_to_tris(connect: np.ndarray) -> np.ndarray:
    """
    Split each quad element into 2 triangles for matplotlib triplot.

    Quad nodes [0,1,2,3] -> triangles [0,1,2] and [0,2,3].
    """
    tris = []
    for quad in connect:
        n0, n1, n2, n3 = quad
        tris.append([n0, n1, n2])
        tris.append([n0, n2, n3])
    return np.array(tris, dtype=int)


# ---------------------------------------------------------------------------
# Helper: plot a nodal field on a quad mesh
# ---------------------------------------------------------------------------

def _plot(coords: np.ndarray, connect: np.ndarray, values: np.ndarray, title: str) -> None:
    """Plot nodal field values on a quad mesh by splitting into triangles."""
    tris = _quads_to_tris(connect)
    felib.plotting.tplot(coords, tris, values, title=title)


# ---------------------------------------------------------------------------
# Helper: make a CPnT4S element with given properties
# ---------------------------------------------------------------------------

def _make_element(alpha: float = 1.2e-5, T_ref: float = 0.0) -> CPnT4S:
    el = CPnT4S()
    el.alpha = alpha
    el.T_ref = T_ref
    return el


def _make_material(E: float, nu: float, k: float = 50.0) -> felib.material.LinearElastic:
    mat = felib.material.LinearElastic(density=1.0, youngs_modulus=E, poissons_ratio=nu)
    mat.conductivity = k
    return mat


# ---------------------------------------------------------------------------
# Helper: extract nodal field for a given DOF constant
# ---------------------------------------------------------------------------

def _get_field(simulation, model, dof_const: int) -> np.ndarray:
    col = model.node_freedom_cols[dof_const]
    vals = np.zeros(model.nnode)
    for lid in range(model.nnode):
        gdof = model.dof_map[lid, col]
        vals[lid] = simulation.dofs[1][gdof]
    return vals


# ---------------------------------------------------------------------------
# Shared mesh: 2x2 quad grid on [-1,1]^2
# ---------------------------------------------------------------------------

def _make_mesh() -> tuple:
    nodes = [
        [1, -1.0, -1.0], [2,  0.0, -1.0], [3,  1.0, -1.0],
        [4, -1.0,  0.0], [5,  0.0,  0.0], [6,  1.0,  0.0],
        [7, -1.0,  1.0], [8,  0.0,  1.0], [9,  1.0,  1.0],
    ]
    elements = [
        [1, 1, 2, 5, 4], [2, 2, 3, 6, 5],
        [3, 4, 5, 8, 7], [4, 5, 6, 9, 8],
    ]
    return nodes, elements


# ---------------------------------------------------------------------------
# Case 1: Uniform temperature plate with clamped edges
# ---------------------------------------------------------------------------

def case1_uniform_temperature() -> None:
    """
    Clamped plate under uniform temperature rise dT = 100 K.

    Expected:
    - All displacements zero (fully clamped)
    - Thermal stresses: sigma_xx = sigma_yy = -E*alpha*dT/(1-nu)
    """
    print("\n=== Case 1: Uniform Temperature Plate (Clamped) ===")

    nodes, elements = _make_mesh()

    mesh = felib.mesh.Mesh(nodes, elements)
    mesh.block(name="Block-1", elements=[1, 2, 3, 4], cell_type=felib.element.Quad4)
    mesh.nodeset(name="boundary", nodes=[1, 2, 3, 4, 6, 7, 8, 9])

    E, nu, alpha, dT = 200e9, 0.3, 1.2e-5, 100.0

    mat = _make_material(E, nu)
    el  = _make_element(alpha=alpha, T_ref=0.0)

    model = felib.model.Model(mesh, name="uniform_temp_plate")
    model.assign_properties(block="Block-1", element=el, material=mat)

    simulation = felib.simulation.Simulation(model)
    step = simulation.staggered_step(name="thermo-mech")
    step.thermal.temperature(nodes="boundary", value=dT)
    step.mechanical.boundary(nodes="boundary", dofs=[X, Y], value=0.0)
    simulation.run()

    T_vals  = _get_field(simulation, model, felib.constants.T)
    ux_vals = _get_field(simulation, model, felib.constants.Ux)
    uy_vals = _get_field(simulation, model, felib.constants.Uy)

    sigma_expected = -E * alpha * dT / (1 - nu)
    print(f"  Temperature range : {T_vals.min():.1f} - {T_vals.max():.1f} K")
    print(f"  Max |Ux|          : {np.max(np.abs(ux_vals)):.2e} m  (expected ~0)")
    print(f"  Max |Uy|          : {np.max(np.abs(uy_vals)):.2e} m  (expected ~0)")
    print(f"  Expected sigma_xx : {sigma_expected:.3e} Pa")

    _plot(model.coords, model.connect, T_vals,  "Case 1: Temperature Field (Uniform dT=100K)")
    _plot(model.coords, model.connect, ux_vals, "Case 1: Ux Displacement (Clamped, expect ~0)")
    _plot(model.coords, model.connect, uy_vals, "Case 1: Uy Displacement (Clamped, expect ~0)")


# ---------------------------------------------------------------------------
# Case 2: Linear temperature gradient -> bending
# ---------------------------------------------------------------------------

def case2_linear_gradient() -> None:
    """
    Free plate with linear gradient: T=-50 on left, T=+50 on right.

    Expected:
    - Non-uniform expansion -> plate displaces/bends in x direction
    """
    print("\n=== Case 2: Linear Temperature Gradient (Free Plate) ===")

    nodes, elements = _make_mesh()

    mesh = felib.mesh.Mesh(nodes, elements)
    mesh.block(name="Block-1", elements=[1, 2, 3, 4], cell_type=felib.element.Quad4)
    mesh.nodeset(name="left",   nodes=[1, 4, 7])
    mesh.nodeset(name="right",  nodes=[3, 6, 9])
    mesh.nodeset(name="corner", nodes=[1])

    mat = _make_material(E=200e9, nu=0.3)
    el  = _make_element(alpha=1.2e-5, T_ref=0.0)

    model = felib.model.Model(mesh, name="gradient_plate")
    model.assign_properties(block="Block-1", element=el, material=mat)

    simulation = felib.simulation.Simulation(model)
    step = simulation.staggered_step(name="thermo-mech")
    step.thermal.temperature(nodes="left",  value=-50.0)
    step.thermal.temperature(nodes="right", value=50.0)
    step.mechanical.boundary(nodes="corner", dofs=[X, Y], value=0.0)
    simulation.run()

    T_vals  = _get_field(simulation, model, felib.constants.T)
    ux_vals = _get_field(simulation, model, felib.constants.Ux)
    uy_vals = _get_field(simulation, model, felib.constants.Uy)

    print(f"  Temperature range : {T_vals.min():.1f} - {T_vals.max():.1f} K")
    print(f"  Max |Ux|          : {np.max(np.abs(ux_vals)):.4e} m")
    print(f"  Max |Uy|          : {np.max(np.abs(uy_vals)):.4e} m")

    _plot(model.coords, model.connect, T_vals,  "Case 2: Temperature Field (Linear Gradient)")
    _plot(model.coords, model.connect, ux_vals, "Case 2: Ux Displacement (Thermal Bending)")
    _plot(model.coords, model.connect, uy_vals, "Case 2: Uy Displacement (Thermal Bending)")


# ---------------------------------------------------------------------------
# Case 3: Multi-material block
# ---------------------------------------------------------------------------

def case3_multi_material() -> None:
    """
    Two-material block under uniform temperature rise dT = 100 K.

    Left half:  Steel     (E=200 GPa, alpha=1.2e-5)
    Right half: Aluminium (E=70 GPa,  alpha=2.3e-5)

    Expected:
    - Different CTEs -> stress concentration at interface
    - Al expands more -> tension in steel, compression in Al
    """
    print("\n=== Case 3: Multi-Material Block ===")

    nodes, elements = _make_mesh()

    mesh = felib.mesh.Mesh(nodes, elements)
    mesh.block(name="Steel",     elements=[1, 3], cell_type=felib.element.Quad4)
    mesh.block(name="Aluminium", elements=[2, 4], cell_type=felib.element.Quad4)
    mesh.nodeset(name="all_nodes", nodes=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    mesh.nodeset(name="bottom",    nodes=[1, 2, 3])
    mesh.nodeset(name="corner",    nodes=[1])

    dT = 100.0

    mat_steel = _make_material(E=200e9, nu=0.3,  k=50.0)
    mat_alum  = _make_material(E=70e9,  nu=0.33, k=200.0)
    el_steel  = _make_element(alpha=1.2e-5, T_ref=0.0)
    el_alum   = _make_element(alpha=2.3e-5, T_ref=0.0)

    model = felib.model.Model(mesh, name="multi_material_block")
    model.assign_properties(block="Steel",     element=el_steel, material=mat_steel)
    model.assign_properties(block="Aluminium", element=el_alum,  material=mat_alum)

    simulation = felib.simulation.Simulation(model)
    step = simulation.staggered_step(name="thermo-mech")
    step.thermal.temperature(nodes="all_nodes", value=dT)
    step.mechanical.boundary(nodes="bottom", dofs=[Y], value=0.0)
    step.mechanical.boundary(nodes="corner", dofs=[X], value=0.0)
    simulation.run()

    T_vals  = _get_field(simulation, model, felib.constants.T)
    ux_vals = _get_field(simulation, model, felib.constants.Ux)
    uy_vals = _get_field(simulation, model, felib.constants.Uy)

    print(f"  Temperature range : {T_vals.min():.1f} - {T_vals.max():.1f} K")
    print(f"  Max |Ux|          : {np.max(np.abs(ux_vals)):.4e} m")
    print(f"  Max |Uy|          : {np.max(np.abs(uy_vals)):.4e} m")
    print(f"  Steel     alpha={el_steel.alpha:.2e}, E={mat_steel.youngs_modulus:.2e} Pa")
    print(f"  Aluminium alpha={el_alum.alpha:.2e},  E={mat_alum.youngs_modulus:.2e} Pa")
    print("  -> Higher CTE in Al causes larger expansion on right side")

    _plot(model.coords, model.connect, T_vals,  "Case 3: Temperature (Multi-Material, dT=100K)")
    _plot(model.coords, model.connect, ux_vals, "Case 3: Ux Displacement (Steel | Aluminium)")
    _plot(model.coords, model.connect, uy_vals, "Case 3: Uy Displacement (Steel | Aluminium)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="CPnT verification examples")
    p.add_argument(
        "--case", type=int, choices=[1, 2, 3], default=0,
        help="Which case to run (1, 2, or 3). Default: all."
    )
    args = p.parse_args()

    if args.case == 0 or args.case == 1:
        case1_uniform_temperature()
    if args.case == 0 or args.case == 2:
        case2_linear_gradient()
    if args.case == 0 or args.case == 3:
        case3_multi_material()

    return 0


if __name__ == "__main__":
    sys.exit(main())