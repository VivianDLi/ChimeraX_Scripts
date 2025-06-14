from typing import List, Tuple

import numpy as np
from scipy import spatial
from sklearn.cluster import OPTICS, cluster_optics_dbscan

from chimerax.core.commands import CmdDesc, SurfaceArg, FloatArg, BoolArg
from chimerax.map import Volume, VolumeSurface

# ============================================================================
# Functions and descriptions for registering using ChimeraX bundle API
# ============================================================================


def volume_distance(
    session,
    source: VolumeSurface,
    to: VolumeSurface | None = None,
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
                session, source, radius=radius, use_internal=use_internal
            )
        else:
            distances = _calculate_intervolume_distance(
                session, source, to, radius=radius, use_internal=use_internal
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
    required=[("source", SurfaceArg)],
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
    cluster_arrays = []
    for cluster_id in cluster_ids:
        if cluster_id == -1:
            # Skip noise points
            continue
        mask = clusters == cluster_id
        cluster_points = points[mask]
        cluster_arrays.append(cluster_points)
    return cluster_arrays


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
    # Create KDTree for fast nearest neighbour search
    tree = spatial.KDTree(surface_points)
    distance, closest_idx = tree.query(point, workers=-1)
    if distance == np.inf or closest_idx == tree.n:  # No neighbours found
        return None, None
    closest_point = tree.data[closest_idx]
    return closest_point, distance


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
    # Setup psudobonds to visualize distances
    pseudobond_grp = session.pb_manager.get_group("distances", create=True)
    if pseudobond_grp.id is None:
        session.models.add([pseudobond_grp])
    pseudobond_grp.dashes = 6
    session.pb_dist_monitor.add_group(pseudobond_grp)
    # Display distances between markers
    for i in range(len(source_markers)):
        source_marker = source_markers[i]
        surface_marker = surface_markers[i]
        pb = pseudobond_grp.new_pseudobond(source_marker, surface_marker)
        pb.color = color
        pb.radius = radius


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
                point, surface_points_outside, max_distance=np.inf
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
