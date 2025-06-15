from typing import List, Tuple

import numpy as np
from scipy import spatial
from sklearn.cluster import OPTICS, cluster_optics_dbscan

from chimerax.core.commands import CmdDesc, ModelArg, FloatArg, BoolArg
from chimerax.core.models import Model
from chimerax.markers import MarkerSet, create_link
from chimerax.map import MapArg, MapsArg, Volume

# ============================================================================
# Functions and descriptions for registering using ChimeraX bundle API
# ============================================================================


def volume_distance_single(
    session,
    source: Volume,
    to: Volume | None = None,
    radius: float = 1.0,
    use_surface: bool = True,
    use_internal: bool = False,
    use_mean: bool = True,
    time_it: bool = True,
) -> None:
    """Calculate distances from a segmented volume/surface to the closest other segmented volume/surface and displays them."""
    session.logger.info("Loading volume data...")
    try:
        if time_it:
            import time

            start_time = time.time()
        session.logger.info(
            f"Calculating distances from source '{source.name}' to target '{to.name if to else 'self'}' with radius {radius}, use_surface={use_surface}, and use_internal={use_internal}."
        )
        if to is None:
            distances = _calculate_intravolume_distance(
                session,
                [source],
                radius=radius,
                use_surface=use_surface,
                use_internal=use_internal,
                use_mean=use_mean,
            )
        else:
            distances = _calculate_intervolume_distance(
                session,
                [source],
                [to],
                radius=radius,
                use_surface=use_surface,
                use_internal=use_internal,
                use_mean=use_mean,
            )
        session.logger.info(f"Distance calculation completed.")
        if time_it:
            end_time = time.time()
            session.logger.info(
                f"Total time taken: {end_time - start_time:.2f} seconds."
            )
        # Save distances to a file
        session.logger.info(f"Distances: {distances}")
        with open("distances.txt", "w") as f:
            for distance in distances:
                f.write(f"{distance}\n")
    except Exception as e:
        import traceback

        session.logger.error(f"Error calculating distances: {e}")
        session.logger.info(traceback.format_exc())


volume_distance_single_desc = CmdDesc(
    required=[("source", MapArg)],
    keyword=[
        ("to", MapArg),
        ("radius", FloatArg),
        ("use_surface", BoolArg),
        ("use_internal", BoolArg),
        ("use_mean", BoolArg),
        ("time_it", BoolArg),
    ],
    synopsis="Calculate the distance from distinct structures in a 3D volume to the closest points in another volume.",
)


def volume_distance_multi(
    session,
    sources: List[Volume],
    tos: List[Volume] | None = None,
    radius: float = 1.0,
    use_surface: bool = True,
    use_internal: bool = False,
    use_mean: bool = True,
    time_it: bool = True,
) -> None:
    """Calculate distances from a segmented volume/surface to the closest other segmented volume/surface and displays them."""
    session.logger.info("Loading volume data...")
    try:
        if time_it:
            import time

            start_time = time.time()
        session.logger.info(
            f"Calculating distances from sources '{[source.name for source in sources]}' to targets '{[to.name for to in tos] if tos else 'self'}' with radius {radius}, use_surface={use_surface}, and use_internal={use_internal}."
        )
        if tos is None:
            distances = _calculate_intravolume_distance(
                session,
                sources,
                radius=radius,
                use_surface=use_surface,
                use_internal=use_internal,
                use_mean=use_mean,
            )
        else:
            distances = _calculate_intervolume_distance(
                session,
                sources,
                tos,
                radius=radius,
                use_surface=use_surface,
                use_internal=use_internal,
                use_mean=use_mean,
            )
        session.logger.info(f"Distance calculation completed.")
        if time_it:
            end_time = time.time()
            session.logger.info(
                f"Total time taken: {end_time - start_time:.2f} seconds."
            )
        # Save distances to a file
        session.logger.info(f"Distances: {distances}")
        with open("distances.txt", "w") as f:
            for distance in distances:
                f.write(f"{distance}\n")
    except Exception as e:
        import traceback

        session.logger.error(f"Error calculating distances: {e}")
        session.logger.info(traceback.format_exc())


volume_distance_multi_desc = CmdDesc(
    required=[("sources", MapsArg)],
    keyword=[
        ("tos", MapsArg),
        ("radius", FloatArg),
        ("use_surface", BoolArg),
        ("use_internal", BoolArg),
        ("use_mean", BoolArg),
        ("time_it", BoolArg),
    ],
    synopsis="Calculate the distance from distinct structures in 3D volumes to the closest points in other volumes.",
)


def volume_distance_group(
    session,
    source: Model,
    to: Model | None = None,
    radius: float = 1.0,
    use_surface: bool = True,
    use_internal: bool = False,
    use_mean: bool = True,
    time_it: bool = True,
) -> None:
    """Calculate distances from a segmented volume/surface to the closest other segmented volume/surface and displays them."""
    session.logger.info("Loading volume data...")
    try:
        if time_it:
            import time

            start_time = time.time()
        session.logger.info(
            f"Calculating distances from source group '{source.name}' to target group '{to.name if to else 'self'}' with radius {radius}, use_surface={use_surface}, and use_internal={use_internal}."
        )
        source_volumes = [m for m in source.child_models() if isinstance(m, Volume)]
        if len(source_volumes) == 0:
            session.logger.info("Couldn't find any volumes inside source group.")
            return
        session.logger.info("Found %d volumes in source group.", len(source_volumes))
        if to is None:
            distances = _calculate_intravolume_distance(
                session,
                source_volumes,
                radius=radius,
                use_internal=use_internal,
                use_mean=use_mean,
            )
        else:
            target_volumes = [m for m in to.child_models() if isinstance(m, Volume)]
            if len(target_volumes) == 0:
                session.logger.info("Couldn't find any volumes inside target group.")
                return
            distances = _calculate_intervolume_distance(
                session,
                source_volumes,
                target_volumes,
                radius=radius,
                use_internal=use_internal,
                use_mean=use_mean,
            )
        session.logger.info(f"Distance calculation completed.")
        if time_it:
            end_time = time.time()
            session.logger.info(
                f"Total time taken: {end_time - start_time:.2f} seconds."
            )
        # Save distances to a file
        session.logger.info(f"Distances: {distances}")
        with open("distances.txt", "w") as f:
            for distance in distances:
                f.write(f"{distance}\n")
    except Exception as e:
        import traceback

        session.logger.error(f"Error calculating distances: {e}")
        session.logger.info(traceback.format_exc())


volume_distance_group_desc = CmdDesc(
    required=[("source", ModelArg)],
    keyword=[
        ("to", ModelArg),
        ("radius", FloatArg),
        ("use_surface", BoolArg),
        ("use_internal", BoolArg),
        ("use_mean", BoolArg),
        ("time_it", BoolArg),
    ],
    synopsis="Calculate the distance from distinct structures in a group of 3D volumes to the closest points in another group of volumes.",
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
        nbrhood_cov = np.cov(nbrhood_centered, rowvar=False, dtype=np.float32)
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


def _calculate_distances_between_clusters(
    session,
    source_clusters: List[np.ndarray],
    target_clusters: List[np.ndarray],
    use_internal: bool = False,
    use_mean: bool = True,
) -> np.ndarray:
    """Calculate distances between clusters of points."""
    # Calculate distances between clusters
    source_points = np.zeros((len(source_clusters), 3), dtype=np.float32)
    closest_points = np.zeros((len(source_clusters), 3), dtype=np.float32)
    distances = np.zeros(len(source_clusters), dtype=np.float32)
    for i, s_c in enumerate(source_clusters):
        # Remove overlapping clusters
        separate_target_clusters = [
            t_c for t_c in target_clusters if not np.array_equal(s_c, t_c)
        ]
        target_surface_points = (
            separate_target_clusters[0]
            if len(separate_target_clusters) == 1
            else np.vstack(separate_target_clusters)
        )
        if use_mean:  # Use source surface mean as the point for distance calculations
            point = np.mean(s_c, axis=0)
            if use_internal:
                # Use ChimeraX find_closest_points method
                closest_point, distance = _calculate_distance_internal(
                    point, target_surface_points, max_distance=np.inf
                )
            else:
                # Use octree method
                closest_point, distance = _calculate_distance(
                    point, target_surface_points
                )
        else:  # Check every source point for the shortest surface-to-surface distance
            point = None
            shortest_distance = np.inf
            for point in s_c:
                if use_internal:
                    # Use ChimeraX find_closest_points method
                    closest_point, distance = _calculate_distance_internal(
                        point, target_surface_points, max_distance=np.inf
                    )
                else:
                    # Use octree method
                    closest_point, distance = _calculate_distance(
                        point, target_surface_points
                    )
                if distance < shortest_distance:
                    shortest_distance = distance
                    point = point
        if closest_point is None or distance is None:
            session.logger.warning(
                f"Could not calculate distance for the cluster {i}'."
            )
            continue
        else:
            source_points[i] = point
            closest_points[i] = closest_point
            distances[i] = distance
    # Remove zero-distance clusters
    source_points = source_points[~(distances == 0)]
    closest_points = closest_points[~(distances == 0)]
    distances = distances[~(distances == 0)]
    return source_points, closest_points, distances


def _display_distance(
    session,
    source_points: np.ndarray,
    closest_points: np.ndarray,
    distances: np.array,
    radius: float = 0.5,
    color: tuple = (255, 255, 0, 255),
) -> None:
    """Display the distance between two points in ChimeraX."""
    from chimerax.label.label3d import label
    from chimerax.core.colors import Color
    from chimerax.core.objects import Objects
    from chimerax.atomic import Bonds

    # Ensure source_points and closest_points are the same length
    if len(source_points) != len(closest_points):
        session.logger.error(
            "Source points, closest points, and distances must have the same length."
        )
        return
    session.logger.info(f"Displaying {len(source_points)} distances.")
    # Define markers to visualize the distances
    endpoint_marker_set = MarkerSet(session, name="endpoints")
    session.models.add([endpoint_marker_set])
    source_markers = [
        endpoint_marker_set.create_marker(source_point, color, radius)
        for source_point in source_points
    ]
    surface_markers = [
        endpoint_marker_set.create_marker(closest_point, color, radius)
        for closest_point in closest_points
    ]
    # Display distances between markers
    for source_marker, surface_marker, distance in zip(
        source_markers, surface_markers, distances
    ):
        link = create_link(source_marker, surface_marker, rgba=color, radius=radius)
        b = Objects(bonds=Bonds([link]))
        label(
            session,
            objects=b,
            object_type="bonds",
            text=str(distance),
            height=250,
            color=Color(color),
        )


# ============================================================================
# Internal functions for running commands
# ============================================================================


def _calculate_intravolume_distance(
    session,
    sources: List[Volume],
    radius: float = 1.0,
    use_surface: bool = True,
    use_internal: bool = False,
    use_mean: bool = True,
) -> np.array:
    """Calculate distances from a segmented volume/surface to the closest points in itself and displays them."""
    if use_surface:
        surfaces = [s.surfaces[0] for s in sources if len(s.surfaces) > 0]
        if len(surfaces) != len(sources):
            session.logger.info(
                f"{len(sources) - len(surfaces)} volumes are missing surfaces and have been skipped."
            )
        surface_points = [
            np.array(surf.vertices).astype(np.float32) for surf in surfaces
        ]
        # Increase neighbourhood radius size for surface points
        radius = 5 * radius
    else:
        # Convert volume data into surface points
        surface_points = []
        for volume in sources:
            data_matrix = volume.full_matrix()
            source_idx = np.argwhere(data_matrix > 0).astype(
                np.uint16
            )  # should be the same, just translated and rotated
            if source_idx.size == 0:
                raise ValueError(
                    f"Source volume {volume.name} does not contain any non-zero data."
                )
            surface_points.append(
                _calculate_surface_points(source_idx, radius=radius).astype(np.float32)
            )
    # Cluster points into distinct structures if single volume
    if len(surface_points) == 1:
        surface_points = surface_points[0]
        surface_points = _separate_structures(surface_points, radius=radius)
    # Calculate distances between clusters
    source_points, closest_points, distances = _calculate_distances_between_clusters(
        session,
        surface_points,
        [np.copy(surf) for surf in surface_points],
        use_internal=use_internal,
        use_mean=use_mean,
    )
    # Display distance
    _display_distance(session, source_points, closest_points, distances)
    return distances


def _calculate_intervolume_distance(
    session,
    sources: List[Volume],
    targets: List[Volume],
    radius: float = 1.0,
    use_surface: bool = True,
    use_internal: bool = False,
    use_mean: bool = True,
) -> np.array:
    """Calculate distances from a segmented volume/surface to the closest other segmented surface and displays them."""
    if use_surface:
        source_surfaces = [s.surfaces[0] for s in sources if len(s.surfaces) > 0]
        target_surfaces = [t.surfaces[0] for t in targets if len(t.surfaces) > 0]
        if len(source_surfaces) != len(sources):
            session.logger.info(
                f"{len(sources) - len(source_surfaces)} source volumes are missing surfaces and have been skipped."
            )
        if len(target_surfaces) != len(targets):
            session.logger.info(
                f"{len(targets) - len(target_surfaces)} target volumes are missing surfaces and have been skipped."
            )
        source_surface_points = [
            np.array(surf.vertices).astype(np.float32) for surf in source_surfaces
        ]
        target_surface_points = [
            np.array(surf.vertices).astype(np.float32) for surf in target_surfaces
        ]
        # Increase neighbourhood radius size for surface points
        radius = 5 * radius
    else:
        # Convert volume data into surface points
        source_surface_points = []
        target_surface_points = []
        for volume in sources:
            data_matrix = volume.full_matrix()
            source_idx = np.argwhere(data_matrix > 0).astype(
                np.uint16
            )  # should be the same, just translated and rotated
            if source_idx.size == 0:
                raise ValueError(
                    f"Source volume {volume.name} does not contain any non-zero data."
                )
            source_surface_points.append(
                _calculate_surface_points(source_idx, radius=radius).astype(np.float32)
            )
        for volume in targets:
            data_matrix = volume.full_matrix()
            target_idx = np.argwhere(data_matrix > 0).astype(np.uint16)
            if target_idx.size == 0:
                raise ValueError(
                    f"Target volume {volume.name} does not contain any non-zero data."
                )
            target_surface_points.append(
                _calculate_surface_points(target_idx, radius=radius).astype(np.float32)
            )
    # Cluster points into distinct structures if single volume
    if len(source_surface_points) == 1:
        source_surface_points = source_surface_points[0]
        source_surface_points = _separate_structures(
            source_surface_points, radius=radius
        )
    # Calculate distances between cluster and target surface points
    source_points, closest_points, distances = _calculate_distances_between_clusters(
        session,
        source_surface_points,
        target_surface_points,
        use_internal=use_internal,
        use_mean=use_mean,
    )
    # Display distance
    _display_distance(session, source_points, closest_points, distances)
    return distances
