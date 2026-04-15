import inspect
import numpy as np
import pytest

import felib


X, Y = felib.X, felib.Y


def build_patch8_mesh():
    """
    2x2 square patch split into 4 Quad8 elements.

    Node layout over [0, 1] x [0, 1]:
        1   2   3   4   5
        6       7       8
        9  10  11  12  13
       14      15      16
       17  18  19  20  21

    The interior node is 11.
    """
    nodes = [
        [1, 0.0, 0.0],
        [2, 0.25, 0.0],
        [3, 0.50, 0.0],
        [4, 0.75, 0.0],
        [5, 1.00, 0.0],
        [6, 0.0, 0.25],
        [7, 0.50, 0.25],
        [8, 1.00, 0.25],
        [9, 0.0, 0.50],
        [10, 0.25, 0.50],
        [11, 0.50, 0.50],
        [12, 0.75, 0.50],
        [13, 1.00, 0.50],
        [14, 0.0, 0.75],
        [15, 0.50, 0.75],
        [16, 1.00, 0.75],
        [17, 0.0, 1.00],
        [18, 0.25, 1.00],
        [19, 0.50, 1.00],
        [20, 0.75, 1.00],
        [21, 1.00, 1.00],
    ]

    # Standard Quad8 ordering:
    # 1 bottom-left, 2 bottom-right, 3 top-right, 4 top-left,
    # 5 mid-bottom, 6 mid-right, 7 mid-top, 8 mid-left
    elements = [
        [1, 1, 3, 11, 9, 2, 7, 10, 6],     # lower-left
        [2, 3, 5, 13, 11, 4, 8, 12, 7],    # lower-right
        [3, 9, 11, 19, 17, 10, 15, 18, 14],# upper-left
        [4, 11, 13, 21, 19, 12, 16, 20, 15],# upper-right
    ]

    mesh = felib.mesh.Mesh(nodes, elements)
    mesh.block(name="All", elements=[1, 2, 3, 4], cell_type=felib.element.Quad8)

    # Outer boundary nodes only for the Y=0 clamp.
    mesh.nodeset(name="YFIX", nodes=[1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21])

    # X displacement groups on the boundary.
    mesh.nodeset(name="X00", nodes=[1, 6, 9, 14, 17])
    mesh.nodeset(name="X025", nodes=[2, 10, 18])
    mesh.nodeset(name="X05", nodes=[3, 19])   # boundary nodes at x = 0.5
    mesh.nodeset(name="X075", nodes=[4, 12, 20])
    mesh.nodeset(name="X10", nodes=[5, 8, 13, 16, 21])

    return mesh, nodes


def _apply_prescribed_bc(step, *, nodes, dofs, value):
    """
    Apply a boundary condition using the available API keyword if present.
    """
    sig = inspect.signature(step.boundary)
    params = sig.parameters

    kwargs = {"nodes": nodes, "dofs": dofs}

    # Most likely names for a prescribed value argument.
    for key in ("value", "values", "u", "displacement", "components", "prescribed", "magnitude"):
        if key in params:
            kwargs[key] = value
            step.boundary(**kwargs)
            return

    # Zero-valued constraints are usually allowed without a special keyword.
    if value == 0.0:
        step.boundary(**kwargs)
        return

    raise RuntimeError(
        "step.boundary(...) does not appear to accept a nonzero prescribed-value keyword. "
        "Please send the boundary method definition and I will adapt this test exactly."
    )


def final_nodal_displacements(simulation):
    """
    Return the final nodal displacement field as (nnodes, 2).
    """
    data = np.asarray(simulation.ndata.data)

    if data.ndim == 3:
        return data[-1, :, :2]
    if data.ndim == 2:
        return data[:, :2]

    raise ValueError(f"Unexpected ndata.data shape: {data.shape}")


def run_patch8(element_obj, stretch=0.5):
    mesh, nodes = build_patch8_mesh()

    model = felib.model.Model(mesh, name=f"patch8_{element_obj.__class__.__name__}")
    material = felib.material.LinearElastic(
        density=1.0,
        youngs_modulus=1.0e03,
        poissons_ratio=2.5e-01,
    )

    model.assign_properties(block="All", element=element_obj, material=material)

    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step()

    # Affine displacement field:
    #   u_x = stretch * x
    #   u_y = 0
    _apply_prescribed_bc(step, nodes="YFIX", dofs=[Y], value=0.0)

    _apply_prescribed_bc(step, nodes="X00", dofs=[X], value=0.0)
    _apply_prescribed_bc(step, nodes="X025", dofs=[X], value=stretch * 0.25)
    _apply_prescribed_bc(step, nodes="X05", dofs=[X], value=stretch * 0.50)
    _apply_prescribed_bc(step, nodes="X075", dofs=[X], value=stretch * 0.75)
    _apply_prescribed_bc(step, nodes="X10", dofs=[X], value=stretch * 1.00)

    simulation.run()

    u = final_nodal_displacements(simulation)
    x = np.array([x for _, x, _ in nodes], dtype=float)
    expected_ux = stretch * x
    expected_uy = np.zeros_like(expected_ux)

    assert np.allclose(u[:, 0], expected_ux, atol=1e-12, rtol=1e-12)
    assert np.allclose(u[:, 1], expected_uy, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "element_factory",
    [
        felib.element.CPE8,
        felib.element.CPE8NL,
        # If you want plane stress instead, swap to:
        # felib.element.CPS8,
        # felib.element.CPS8NL,
    ],
)
def test_patch8(element_factory):
    run_patch8(element_factory())


if __name__ == "__main__":
    run_patch8(felib.element.CPE8())
    run_patch8(felib.element.CPE8NL())
    print("patch8 passed")