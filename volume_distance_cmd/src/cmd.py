from queue import PriorityQueue
from typing import List, Tuple
from .octree import Point, OcTree

import numpy as np
from scipy import spatial
from sklearn.cluster import OPTICS, cluster_optics_dbscan

from chimerax.core.commands import CmdDesc, ModelsArg, SurfaceArg, FloatArg, BoolArg
from chimerax.map import Volume, VolumeSurface

# ============================================================================
# Functions and descriptions for registering using ChimeraX bundle API
# ============================================================================


def volume_distance(
    session,
    source: Volume | VolumeSurface,
    to: VolumeSurface | None,
    radius: float = 1.0,
    use_internal: bool = False,
    time_it: bool = True,
) -> None:
    """Calculate distances from a segmented volume/surface to the closest other segmented volume/surface and displays them."""
    if not isinstance(source, (Volume, VolumeSurface)):
        session.logger.error("Source must be a Volume or VolumeSurface.")
        return
    session.logger.info("Loading volume data...")
    try:
        if time_it:
            import time

            start_time = time.time()
        session.logger.info(
            f"Calculating distances from source '{source.name}' to target '{to.name if to else 'self'}' with radius {radius} and internal={use_internal}."
        )
        if to is None:
            distances = _calculate_intravolume_distance(
                session, source, radius=radius, internal=use_internal
            )
        else:
            distances = _calculate_intervolume_distance(
                session, source, to, radius=radius, internal=use_internal
            )
        session.logger.info(f"Distance calculation completed.")
        if time_it:
            end_time = time.time()
            session.logger.info(
                f"Total time taken: {end_time - start_time:.2f} seconds."
            )
        # Save distances to a file
        np.save("distances.npy", distances)
    except Exception as e:
        session.logger.error(f"Error calculating distances: {e}")


volume_distance_desc = CmdDesc(
    required=[("source", ModelsArg)],
    keyword=[
        ("to", SurfaceArg),
        ("radius", FloatArg),
        ("use_internal", BoolArg),
        ("time_it", BoolArg),
    ],
    synopsis="Calculate the distance from distinct structures in a 3D volume to the closest points in another volume.",
)

# ============================================================================
# Functions intended for internal use by the bundle
# ============================================================================

#### Define distance calculation functions ####


def _calculate_distance_between_points(point1: Point, point2: Point) -> float:
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(
        (point1.x - point2.x) ** 2
        + (point1.y - point2.y) ** 2
        + (point1.z - point2.z) ** 2
    )


def _calculate_distance_to_node(point: Point, node: OcTree) -> float:
    """Calculate the minimum distance from a point to an octree node."""
    # Check if the point is within the bounds of the node
    if node.is_within_bounds(point):
        return 0.0
    # If the node is empty, return infinity
    if node.point == Point(-1, -1, -1):
        return float("inf")
    # Get the top-left and bottom-right corners of the node
    top_left, bottom_right = node.top_left, node.bottom_right
    # Calculate distances
    dx = max(bottom_right.x - point.x, point.x - top_left.x, 0)
    dy = max(bottom_right.y - point.y, point.y - top_left.y, 0)
    dz = max(bottom_right.z - point.z, point.z - top_left.z, 0)
    # Return the Euclidean distance from the point to the closest point on the node's surface
    return np.sqrt(dx**2 + dy**2 + dz**2)


def _find_closest_point_octree(point: Point, octree: OcTree) -> Point:
    """Find the closest point in the octree to the given point."""
    queue = PriorityQueue()
    queue.put((0, octree))  # Start with the root node and distance 0
    while not queue.empty():
        _, current_node = queue.get()
        # Check for internal node and add children to the queue
        if current_node.point is None:
            for child in current_node.children:
                if child.point is None:
                    # Calculate the distance from the point to the child node
                    distance_to_child = _calculate_distance_to_node(point, child)
                    queue.put((distance_to_child, child))
                elif child.point == Point(-1, -1, -1):
                    # If the child is an empty node, we can skip it
                    continue
                else:
                    # If the child is a leaf node with a point, calculate the distance
                    distance = _calculate_distance(point, child.point)
                    queue.put((distance, child))
        # Return distance if the current node is a leaf node with a point
        elif current_node.point != Point(-1, -1, -1):
            return current_node.point
        else:
            # No empty nodes should be added to the queue
            raise ValueError("Empty node encountered in search queue.")


#### Define ChimeraX commands for calculating distances ####


def _separate_structures(points: np.ndarray, radius: float = 2.0) -> List[np.ndarray]:
    """Separate 3D classifications into distinct structures using OPTICS clustering and returns the center of those clusters."""
    optics = OPTICS(min_samples=5, max_eps=5 * radius)
    optics.fit(points)
    # Extract clusters with DBSCAN
    clusters = cluster_optics_dbscan(
        reachability=optics.reachability_,
        core_distances=optics.core_distances_,
        ordering=optics.ordering_,
        eps=radius,
    ).astype(np.uint8)
    cluster_ids = np.unique(clusters)
    clusters = []
    for cluster_id in cluster_ids:
        if cluster_id == -1:
            # Skip noise points
            continue
        mask = clusters == cluster_id
        cluster_points = points[mask]
        clusters.append(cluster_points)
    return clusters


def _calculate_surface_points(
    points: np.ndarray, radius: float = 1.0, point_tol: int = 2
) -> np.ndarray:
    """Calculate surface points of a 3D volume as an nx3 numpy array using surface normal estimation."""
    # Get points from volume
    tree = spatial.KDTree(points)
    surface_points = []
    for point in points:
        # Find local neighbourhood around query point
        idx = tree.query_ball_point(point, radius, eps=0, p=2)
        nbrhood = points[idx]
        # Estimate normal direction at query point using SVD
        nbrhood_centered = nbrhood - np.mean(nbrhood, axis=0, dtype=np.float32)
        nbrhood_cov = np.cov(nbrhood_centered, dtype=np.float32)
        U, S, _ = np.linalg.svd(nbrhood_cov)
        # Use smallest eigenvector as normal direction
        normal = U[:, np.argmin(S)]
        # Search for points in circular patches in the normal direction
        centers = (point + normal * radius / 2, point - normal * radius / 2)
        for center in centers:
            nbrhood_idx = tree.query_ball_point(center, radius / 2, eps=0, p=2)
            if len(nbrhood_idx) < point_tol:
                surface_points.append(center)
    return np.unique(surface_points, axis=0)


def _calculate_distance(
    point: np.array, surface_points: np.ndarray
) -> Tuple[np.array, float]:
    """Calculate the distance from each point to the closest point in a list of surface points using OcTrees."""
    top_left, bottom_right = Point(*np.min(surface_points, axis=0)), Point(
        *np.max(surface_points, axis=0)
    )
    octree = OcTree(
        point=None, top_left=top_left, bottom_right=bottom_right
    )  # Create an empty octree
    closest_point = _find_closest_point_octree(Point(*point), octree)
    if closest_point is None:
        return None, None
    distance = _calculate_distance_between_points(Point(*point), closest_point)
    return np.array([closest_point.x, closest_point.y, closest_point.z]), distance


def _calculate_distance_internal(
    point: np.array, surface_points: np.ndarray, max_distance: float = 1000.0
) -> Tuple[np.array, float]:
    from chimerax.geometry import find_closest_points

    """Calculate the distance from each point to the closest point in a list of surface points using ChimeraX find_closest_points method."""
    _, _, closest_idxs = find_closest_points(
        np.expand_dims(point, axis=0), surface_points, max_distance=max_distance
    )
    closest_point = surface_points[closest_idxs[0]]
    distance = np.linalg.norm(point - closest_point, axis=1)
    return closest_point, distance


def _display_distance(
    session,
    source_name: str,
    source_points: np.ndarray,
    surface_name: str,
    closest_points: np.ndarray,
    radius: float = 0.5,
    color: tuple = (255, 255, 0, 255),
) -> None:
    """Display the distance between two points in ChimeraX."""
    from chimerax.markers import MarkerSet
    from chimerax.core.colors import Color
    from chimerax.core.commands import run

    # Ensure source_points and closest_points are the same length
    if len(source_points) != len(closest_points):
        session.logger.error(
            "Source points, closest points, and distances must have the same length."
        )
        return
    session.logger.info(
        f"Displaying {len(source_points)} distances from '{source_name}' to '{surface_name}'."
    )
    # Define markers to visualize the distances
    source_marker_set = MarkerSet(session, name=source_name)
    source_markers = [
        source_marker_set.create_marker(source_point, color, radius)
        for source_point in source_points
    ]
    surface_marker_set = MarkerSet(session, name=surface_name)
    surface_markers = [
        surface_marker_set.create_marker(closest_point, color, radius)
        for closest_point in closest_points
    ]
    # Display distances between markers
    for i in range(len(source_markers)):
        source_marker = source_markers[i]
        surface_marker = surface_markers[i]
        run(
            session,
            "distance %s %s" % (source_marker.atomspec, surface_marker.atomspec),
        )


def _calculate_intravolume_distance(
    session,
    source: Volume | VolumeSurface,
    radius: float = 1.0,
    use_internal: bool = False,
) -> np.array:
    """Calculate distances from a segmented volume/surface to the closest points in itself and displays them."""
    if isinstance(source, Volume):
        # Convert volume data into surface points
        data_matrix = source.full_matrix()
        # TODO: compare idx uint16 image with converted to xyz coordinates
        source_idx = np.argwhere(data_matrix > 0).astype(
            np.uint16
        )  # should be the same, just translated and rotated
        if source_idx.size == 0:
            raise ValueError("Source volume does not contain any non-zero data.")
        surface_points = _calculate_surface_points(source_idx, radius=radius).astype(
            np.float32
        )
    else:
        surface_points = np.array(source.vertices).astype(np.float32)
        # Increase neighbourhood radius size for surface points
        radius = 5 * radius
    # Cluster points into distinct structures
    clusters = _separate_structures(surface_points, radius=radius)
    # Calculate distances between clusters
    source_points = np.zeros((len(clusters), 3), dtype=np.float32)
    closest_points = np.zeros((len(clusters), 3), dtype=np.float32)
    distances = np.zeros(len(clusters), dtype=np.float32)
    for i, cluster in enumerate(clusters):
        point = np.mean(cluster, axis=0)
        # Get all surface points except the current cluster
        surface_points_outside = np.hstack(
            [clusters[j] for j in range(len(clusters)) if j != i]
        )
        if use_internal:
            # Use ChimeraX find_closest_points method
            closest_point, distance = _calculate_distance_internal(
                point, surface_points_outside, max_distance=np.max(source.data.size)
            )
        else:
            # Use octree method
            closest_point, distance = _calculate_distance(point, surface_points_outside)
        if closest_point is None or distance is None:
            session.logger.warning(
                f"Could not calculate distance for point {point} in source '{source.name}'."
            )
            continue
        source_points[i] = point
        closest_points[i] = closest_point
        distances[i] = distance
    # Display distance
    _display_distance(session, source.name, source_points, source.name, closest_points)
    return distances


def _calculate_intervolume_distance(
    session,
    source: Volume | VolumeSurface,
    target: VolumeSurface,
    radius: float = 1.0,
    use_internal: bool = False,
) -> np.array:
    """Calculate distances from a segmented volume/surface to the closest other segmented surface and displays them."""
    if isinstance(source, Volume):
        # Convert volume data into surface points
        data_matrix = source.full_matrix()
        source_idx = np.argwhere(data_matrix > 0).astype(
            np.uint16
        )  # should be the same, just translated and rotated
        if source_idx.size == 0:
            raise ValueError("Source volume does not contain any non-zero data.")
        source_surface_points = _calculate_surface_points(
            source_idx, radius=radius
        ).astype(np.float32)
    else:
        source_surface_points = np.array(source.vertices).astype(np.float32)
    target_surface_points = np.array(target.vertices).astype(np.float32)
    # Cluster points into distinct structures
    clusters = _separate_structures(source_surface_points, radius=radius)
    # Calculate distances between cluster and target surface points
    source_points = np.zeros((len(clusters), 3), dtype=np.float32)
    closest_points = np.zeros((len(clusters), 3), dtype=np.float32)
    distances = np.zeros(len(clusters), dtype=np.float32)
    for i, cluster in enumerate(clusters):
        point = np.mean(cluster, axis=0)
        if use_internal:
            # Use ChimeraX find_closest_points method
            closest_point, distance = _calculate_distance_internal(
                point, target_surface_points, max_distance=np.max(source.data.size)
            )
        else:
            # Use octree method
            closest_point, distance = _calculate_distance(point, target_surface_points)
        if closest_point is None or distance is None:
            session.logger.warning(
                f"Could not calculate distance for point {point} in source '{source.name}'."
            )
            continue
        source_points[i] = point
        closest_points[i] = closest_point
        distances[i] = distance
    # Display distance
    _display_distance(session, source.name, source_points, source.name, closest_points)
    return distances
