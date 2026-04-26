import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np

import felib

X = felib.X
Y = felib.Y


def run_example(
    *,
    output: str | Path = "mirrored_tied_contact_displacement.png",
    show: bool = False,
) -> dict:
    E = 100.0
    nu = 0.3
    total_y_force = 10.0
    total_x_force = 2.5

    material = felib.material.LinearElastic(youngs_modulus=E, poissons_ratio=nu)

    tied_mesh, tied_u = _solve_tied_case(material, total_y_force, total_x_force)
    solid_mesh, solid_u = _solve_solid_case(material, total_y_force, total_x_force)

    tied_by_coord = _displacements_by_coord(tied_mesh, tied_u)
    solid_by_coord = _displacements_by_coord(solid_mesh, solid_u)
    if tied_by_coord.keys() != solid_by_coord.keys():
        raise RuntimeError("Tied and solid meshes do not have the same coordinate set")

    coords = sorted(solid_by_coord)
    displacement_error = np.array(
        [tied_by_coord[coord] - solid_by_coord[coord] for coord in coords],
        dtype=float,
    )
    error_norm = np.linalg.norm(displacement_error, axis=1)
    max_error = float(np.max(error_norm))

    output = Path(output)
    _plot_displacement_comparison(
        tied_mesh,
        tied_u,
        solid_mesh,
        solid_u,
        output,
        max_error=max_error,
        show=show,
    )

    print("=== Mirrored tied contact example ===")
    print("Primary mesh: 3 x 4 CPS4 elements, 0.5 x 0.5")
    print("Secondary mesh: mirrored about -y")
    print(f"Maximum tied/solid nodal displacement difference: {max_error:.3e}")
    print(f"Displacement comparison plot: {output}")

    return {
        "tied_mesh": tied_mesh,
        "solid_mesh": solid_mesh,
        "tied_u": tied_u,
        "solid_u": solid_u,
        "error_norm": error_norm,
        "plot": output,
    }


def _solve_tied_case(material, total_y_force: float, total_x_force: float):
    primary_nodes, primary_elems = felib.meshing.rectmesh(
        (0.0, 1.5, 0.0, 2.0),
        nx=3,
        ny=4,
    )
    mirrored = felib.meshing.mirrored_tied_mesh(primary_nodes, primary_elems, "-y")

    mesh = felib.mesh.Mesh(nodes=mirrored.nodes, elements=mirrored.elements)
    mesh.block(name="Primary", cell_type=felib.element.CPS4, elements=mirrored.primary_elements)
    mesh.block(
        name="Secondary",
        cell_type=felib.element.CPS4,
        elements=mirrored.secondary_elements,
    )

    model = felib.model.Model(mesh, name="mirrored_tied_minus_y")
    model.assign_properties(block="Primary", element=felib.element.CPS4(), material=material)
    model.assign_properties(block="Secondary", element=felib.element.CPS4(), material=material)

    secondary_bottom = _nodes_at(mirrored.nodes, y=-2.0)
    primary_top = _nodes_at(
        [node for node in mirrored.nodes if node[0] in mirrored.primary_nodes],
        y=2.0,
    )
    # Interpret "top two nodes" as the two right-most nodes on the primary top edge.
    primary_top_two = _rightmost_nodes(
        [
            node
            for node in mirrored.nodes
            if node[0] in mirrored.primary_nodes and np.isclose(node[2], 2.0)
        ],
        count=2,
    )

    sim = felib.simulation.Simulation(model)
    step = sim.static_step()
    step.boundary(nodes=secondary_bottom, dofs=[X, Y], value=0.0)
    step.point_load(nodes=primary_top, dofs=Y, value=total_y_force / len(primary_top))
    step.point_load(nodes=primary_top_two, dofs=X, value=total_x_force / len(primary_top_two))
    sim.run()

    return mesh, sim.ndata.gather("u")


def _solve_solid_case(material, total_y_force: float, total_x_force: float):
    nodes, elems = felib.meshing.rectmesh(
        (0.0, 1.5, -2.0, 2.0),
        nx=3,
        ny=8,
    )

    mesh = felib.mesh.Mesh(nodes=nodes, elements=elems)
    mesh.block(name="Solid", cell_type=felib.element.CPS4, elements=[elem[0] for elem in elems])

    model = felib.model.Model(mesh, name="mirrored_solid_reference")
    model.assign_properties(block="Solid", element=felib.element.CPS4(), material=material)

    bottom = _nodes_at(nodes, y=-2.0)
    top = _nodes_at(nodes, y=2.0)
    # Apply the x-load at the same coordinates used by the tied primary mesh.
    top_two = _rightmost_nodes([node for node in nodes if np.isclose(node[2], 2.0)], count=2)

    sim = felib.simulation.Simulation(model)
    step = sim.static_step()
    step.boundary(nodes=bottom, dofs=[X, Y], value=0.0)
    step.point_load(nodes=top, dofs=Y, value=total_y_force / len(top))
    step.point_load(nodes=top_two, dofs=X, value=total_x_force / len(top_two))
    sim.run()

    return mesh, sim.ndata.gather("u")


def _nodes_at(nodes, *, y: float) -> list[int]:
    return [node[0] for node in nodes if np.isclose(node[2], y)]


def _rightmost_nodes(nodes, *, count: int) -> list[int]:
    ordered = sorted(nodes, key=lambda node: (node[1], node[2]))
    return [node[0] for node in ordered[-count:]]


def _displacements_by_coord(mesh, u):
    values = {}
    for coord, disp in zip(mesh.coords, u):
        key = tuple(np.round(coord, 12))
        values.setdefault(key, []).append(disp)
    return {key: np.mean(disps, axis=0) for key, disps in values.items()}


def _plot_displacement_comparison(
    tied_mesh,
    tied_u,
    solid_mesh,
    solid_u,
    output: Path,
    *,
    max_error: float,
    show: bool,
) -> None:
    scale = _deformation_scale(tied_mesh.coords, tied_u, solid_u)
    tied_deformed = tied_mesh.coords + scale * tied_u
    solid_deformed = solid_mesh.coords + scale * solid_u

    fig, ax = plt.subplots(figsize=(8, 8))
    _plot_edges(
        ax, solid_mesh.coords, solid_mesh.connect, color="0.78", linestyle=":", label="original"
    )
    _plot_edges(
        ax,
        tied_deformed,
        tied_mesh.connect,
        color="tab:blue",
        linestyle="-",
        label="tied deformed",
    )
    _plot_edges(
        ax,
        solid_deformed,
        solid_mesh.connect,
        color="tab:orange",
        linestyle="--",
        label="solid deformed",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Mirrored tied vs solid displacement (scale={scale:.2g}, max error={max_error:.2e})"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def _deformation_scale(coords, *displacements) -> float:
    span = np.linalg.norm(np.ptp(coords, axis=0))
    max_u = max(float(np.max(np.linalg.norm(u, axis=1))) for u in displacements)
    if max_u == 0.0:
        return 1.0
    return 0.12 * span / max_u


def _plot_edges(ax, coords, connect, *, color: str, linestyle: str, label: str) -> None:
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    seen = set()
    for elem in connect:
        for n0, n1 in edges:
            edge = tuple(sorted((int(elem[n0]), int(elem[n1]))))
            if edge in seen:
                continue
            seen.add(edge)
            pts = coords[[elem[n0], elem[n1]]]
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=1.0,
                label=label,
            )
            label = None


if __name__ == "__main__":
    run_example(show=False)
