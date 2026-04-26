from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TiedMesh:
    nodes: list[list[int | float]]
    elements: list[list[int]]
    node_pairs: list[tuple[int, int]]
    primary_nodes: list[int]
    secondary_nodes: list[int]
    primary_elements: list[int]
    secondary_elements: list[int]


def mirrored_tied_mesh(
    nodes: Sequence[Sequence[int | float]],
    elements: Sequence[Sequence[int]],
    direction: str,
    *,
    tol: float = 1e-12,
) -> TiedMesh:
    """
    Build a two-half mesh by mirroring a primary mesh across one outside edge.

    Parameters
    ----------
    nodes
        Primary mesh nodes as ``[node_id, x, y, ...]`` rows.
    elements
        Primary mesh elements as ``[elem_id, node_id, ...]`` rows.
    direction
        Copy direction. ``"+x"`` mirrors across ``xmax``, ``"-x"`` mirrors
        across ``xmin``, ``"+y"`` mirrors across ``ymax``, and ``"-y"``
        mirrors across ``ymin``.
    tol
        Coordinate matching tolerance used to identify tied node pairs.

    Returns
    -------
    TiedMesh
        Combined primary/secondary mesh data and node pairs in
        ``(secondary_node, primary_node)`` order for ``StaticStep.tie_nodes``.
    """
    direction = _normalize_direction(direction)
    axis = 0 if direction[1] == "x" else 1
    use_max = direction[0] == "+"

    primary_nodes = [list(row) for row in nodes]
    primary_elements = [list(row) for row in elements]
    if not primary_nodes:
        raise ValueError("Expected at least one primary node")
    if not primary_elements:
        raise ValueError("Expected at least one primary element")

    coords = {int(row[0]): np.asarray(row[1:], dtype=float) for row in primary_nodes}
    ndim = len(next(iter(coords.values())))
    if ndim < 2:
        raise ValueError("mirrored_tied_mesh requires at least two coordinate dimensions")
    if axis >= ndim:
        raise ValueError(f"Cannot mirror {ndim}D coordinates in direction {direction!r}")

    max_node_id = max(coords)
    max_elem_id = max(int(row[0]) for row in primary_elements)
    plane = max(coord[axis] for coord in coords.values())
    if not use_max:
        plane = min(coord[axis] for coord in coords.values())

    secondary_id_by_primary = {
        int(row[0]): max_node_id + i + 1 for i, row in enumerate(primary_nodes)
    }
    secondary_nodes: list[list[int | float]] = []
    secondary_coords: dict[int, NDArray] = {}
    for row in primary_nodes:
        primary_gid = int(row[0])
        secondary_gid = secondary_id_by_primary[primary_gid]
        x = coords[primary_gid].copy()
        x[axis] = 2.0 * plane - x[axis]
        secondary_coords[secondary_gid] = x
        secondary_nodes.append([secondary_gid] + x.tolist())

    all_coords = dict(coords)
    all_coords.update(secondary_coords)

    secondary_elements: list[list[int]] = []
    for i, element in enumerate(primary_elements):
        primary_conn = [int(nid) for nid in element[1:]]
        mirrored_conn = [secondary_id_by_primary[nid] for nid in primary_conn]
        mirrored_conn = _preserve_element_orientation(primary_conn, mirrored_conn, all_coords)
        secondary_elements.append([max_elem_id + i + 1] + mirrored_conn)

    node_pairs = matching_node_pairs(
        primary_nodes=primary_nodes,
        secondary_nodes=secondary_nodes,
        tol=tol,
    )

    return TiedMesh(
        nodes=primary_nodes + secondary_nodes,
        elements=primary_elements + secondary_elements,
        node_pairs=node_pairs,
        primary_nodes=[int(row[0]) for row in primary_nodes],
        secondary_nodes=[int(row[0]) for row in secondary_nodes],
        primary_elements=[int(row[0]) for row in primary_elements],
        secondary_elements=[int(row[0]) for row in secondary_elements],
    )


def matching_node_pairs(
    *,
    primary_nodes: Sequence[Sequence[int | float]],
    secondary_nodes: Sequence[Sequence[int | float]],
    tol: float = 1e-12,
) -> list[tuple[int, int]]:
    """Return ``(secondary, primary)`` pairs for nodes with matching coordinates."""
    primary = [(int(row[0]), np.asarray(row[1:], dtype=float)) for row in primary_nodes]
    secondary = [(int(row[0]), np.asarray(row[1:], dtype=float)) for row in secondary_nodes]

    pairs: list[tuple[int, int]] = []
    for secondary_gid, secondary_coord in secondary:
        matches = [
            primary_gid
            for primary_gid, primary_coord in primary
            if np.allclose(secondary_coord, primary_coord, atol=tol, rtol=0.0)
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Secondary node {secondary_gid} matches multiple primary nodes: {matches}"
            )
        if len(matches) == 1 and matches[0] != secondary_gid:
            pairs.append((secondary_gid, matches[0]))
    return pairs


def _normalize_direction(direction: str) -> str:
    value = direction.strip().lower()
    aliases = {
        "+x": "+x",
        "x+": "+x",
        "-x": "-x",
        "x-": "-x",
        "+y": "+y",
        "y+": "+y",
        "-y": "-y",
        "y-": "-y",
    }
    if value not in aliases:
        raise ValueError("direction must be one of '+x', '-x', '+y', or '-y'")
    return aliases[value]


def _preserve_element_orientation(
    primary_conn: Sequence[int],
    mirrored_conn: Sequence[int],
    coords: dict[int, NDArray],
) -> list[int]:
    primary_area = _signed_area(primary_conn, coords)
    mirrored_area = _signed_area(mirrored_conn, coords)
    if abs(primary_area) < 1e-15 or abs(mirrored_area) < 1e-15:
        raise ValueError("Cannot determine orientation for a degenerate element")
    mirrored = list(mirrored_conn)
    if primary_area * mirrored_area < 0.0:
        try:
            order = _reversed_element_order(len(mirrored))
        except KeyError as exc:
            raise ValueError(
                f"Mirrored orientation correction is not implemented for "
                f"{len(mirrored)}-node elements"
            ) from exc
        mirrored = [mirrored[i] for i in order]
    return mirrored


def _signed_area(conn: Sequence[int], coords: dict[int, NDArray]) -> float:
    corner_ix = _corner_indices(len(conn))
    p = np.asarray([coords[int(conn[i])][:2] for i in corner_ix], dtype=float)
    x = p[:, 0]
    y = p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _corner_indices(nnode: int) -> tuple[int, ...]:
    corners = {
        3: (0, 1, 2),
        4: (0, 1, 2, 3),
        6: (0, 1, 2),
        8: (0, 1, 2, 3),
    }
    try:
        return corners[nnode]
    except KeyError as exc:
        raise ValueError(f"Unsupported element with {nnode} nodes") from exc


def _reversed_element_order(nnode: int) -> tuple[int, ...]:
    orders = {
        3: (0, 2, 1),
        4: (0, 3, 2, 1),
        6: (0, 2, 1, 5, 4, 3),
        8: (0, 3, 2, 1, 7, 6, 5, 4),
    }
    return orders[nnode]
