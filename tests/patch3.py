import inspect
import numpy as np
import pytest

import felib


X, Y = felib.X, felib.Y


def build_patch3_mesh():
    """
    2x2 square patch split into 8 Tri3 elements.
    This mirrors the patch4 geometry, but with triangles.
    """
    nodes = [
        [1, 0.0, 0.0],
        [2, 0.5, 0.0],
        [3, 1.0, 0.0],
        [4, 0.0, 0.5],
        [5, 0.5, 0.5],
        [6, 1.0, 0.5],
        [7, 0.0, 1.0],
        [8, 0.5, 1.0],
        [9, 1.0, 1.0],
    ]

    # Split each quad into 2 triangles, counterclockwise ordering.
    elements = [
        [1, 1, 2, 5],
        [2, 1, 5, 4],
        [3, 2, 3, 6],
        [4, 2, 6, 5],
        [5, 4, 5, 8],
        [6, 4, 8, 7],
        [7, 5, 6, 9],
        [8, 5, 9, 8],
    ]

    mesh = felib.mesh.Mesh(nodes, elements)
    mesh.block(name="All", elements=list(range(1, 9)), cell_type=felib.element.Tri3)

    mesh.nodeset(name="LEFT", nodes=[1, 4, 7])
    mesh.nodeset(name="RIGHT", nodes=[3, 6, 9])
    mesh.nodeset(name="BOTTOM", nodes=[1, 2, 3])
    mesh.nodeset(name="TOP", nodes=[7, 8, 9])

    return mesh, nodes


def _apply_prescribed_bc(step, *, nodes, dofs, value):
    """
    Try to apply a prescribed displacement using the step.boundary API
    without guessing a single keyword name.
    """
    sig = inspect.signature(step.boundary)
    params = sig.parameters

    kwargs = {"nodes": nodes, "dofs": dofs}

    for key in ("value", "values", "u", "displacement", "components", "prescribed", "magnitude"):
        if key in params:
            kwargs[key] = value
            step.boundary(**kwargs)
            return

    if value == 0.0:
        step.boundary(**kwargs)
        return

    raise RuntimeError(
        "step.boundary(...) does not appear to accept a nonzero prescribed-value keyword. "
        "Please send the boundary method definition and I will adapt this test exactly."
    )


def run_patch3(element_obj, stretch=0.5):
    mesh, nodes = build_patch3_mesh()

    model = felib.model.Model(mesh, name=f"patch3_{element_obj.__class__.__name__}")
    material = felib.material.LinearElastic(
        density=1.0,
        youngs_modulus=1.0e03,
        poissons_ratio=2.5e-01,
    )

    model.assign_properties(block="All", element=element_obj, material=material)

    simulation = felib.simulation.Simulation(model)
    step = simulation.static_step()

    # Match the affine displacement field u_x = stretch * x, u_y = 0
    _apply_prescribed_bc(step, nodes="LEFT", dofs=[X, Y], value=0.0)
    _apply_prescribed_bc(step, nodes="BOTTOM", dofs=[Y], value=0.0)
    _apply_prescribed_bc(step, nodes="TOP", dofs=[Y], value=0.0)
    _apply_prescribed_bc(step, nodes="RIGHT", dofs=[X], value=stretch)

    simulation.run()

    data = np.asarray(simulation.ndata.data)
    if data.ndim == 3:
        u = data[-1, :, :2]
    elif data.ndim == 2:
        u = data[:, :2]
    else:
        raise ValueError(f"Unexpected ndata.data shape: {data.shape}")

    x = np.array([x for _, x, _ in nodes], dtype=float)
    expected_ux = stretch * x
    expected_uy = np.zeros_like(expected_ux)

    assert np.allclose(u[:, 0], expected_ux, atol=1e-12, rtol=1e-12)
    assert np.allclose(u[:, 1], expected_uy, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "element_factory",
    [
        felib.element.CPE3,
        felib.element.CPE3NL,
        # For plane stress, swap the two lines above to:
        # felib.element.CPS3,
        # felib.element.CPS3NL,
    ],
)
def test_patch3(element_factory):
    run_patch3(element_factory())