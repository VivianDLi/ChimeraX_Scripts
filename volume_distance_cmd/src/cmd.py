from typing import List, Tuple
from queue import SimpleQueue
import math

import numpy as np
from scipy import spatial
import open3d as o3d

from chimerax.core.commands import (
    CmdDesc,
    ModelArg,
    ModelsArg,
    IntArg,
    FloatArg,
    BoolArg,
)
from chimerax.core.models import Model, Surface
from chimerax.markers import MarkerSet, create_link
from chimerax.map import Volume

# ============================================================================
# Functions and descriptions for registering using ChimeraX bundle API
# ============================================================================


def volume_distance_single(
    session,
    source: Model,
    to: Model | None = None,
    surface_radius: float = 1.0,
    surface_point_tol: int = 2,
    cluster_normal_k: int = 50,
    cluster_region_k: int = 30,
    cluster_curv_threshold: float | None = None,
    cluster_angle_threshold: float | None = None,
    bond_radius: float = 50.0,
    use_surface: bool = True,
    use_internal: bool = False,
    use_mean: bool = True,
    time_it: bool = True,
) -> None:
    """Calculate distances from a segmented volume/surface to the closest other segmented volume/surface and displays them."""
    session.logger.info("Loading volume data...")
    # Check types
    if not isinstance(source, (Volume, Surface)):
        session.logger.error(
            f"Source model must be a Volume or Surface, got {type(source)}."
        )
        return
    if to is not None and not isinstance(to, (Volume, Surface)):
        session.logger.error(
            f"Target model must be a Volume or Surface or not specified, got {type(to)}."
        )
        return
    try:
        if time_it:
            import time

            start_time = time.time()
        session.logger.info(
            f"Calculating distances from source '{source.name}' to target '{to.name if to else 'self'}' with use_surface={use_surface} and use_internal={use_internal}."
        )
        if to is None:
            distances = _calculate_intravolume_distance(
                session,
                [source],
                surface_radius=surface_radius,
                surface_point_tol=surface_point_tol,
                cluster_normal_k=cluster_normal_k,
                cluster_region_k=cluster_region_k,
                cluster_curv_threshold=cluster_curv_threshold,
                cluster_angle_threshold=cluster_angle_threshold,
                bond_radius=bond_radius,
                use_surface=use_surface,
                use_internal=use_internal,
                use_mean=use_mean,
            )
        else:
            distances = _calculate_intervolume_distance(
                session,
                [source],
                [to],
                surface_radius=surface_radius,
                surface_point_tol=surface_point_tol,
                cluster_normal_k=cluster_normal_k,
                cluster_region_k=cluster_region_k,
                cluster_curv_threshold=cluster_curv_threshold,
                cluster_angle_threshold=cluster_angle_threshold,
                bond_radius=bond_radius,
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
    required=[("source", ModelArg)],
    keyword=[
        ("to", ModelArg),
        ("surface_radius", FloatArg),
        ("surface_point_tol", IntArg),
        ("cluster_normal_k", IntArg),
        ("cluster_region_k", IntArg),
        ("cluster_curv_threshold", FloatArg),
        ("cluster_angle_threshold", FloatArg),
        ("bond_radius", FloatArg),
        ("use_surface", BoolArg),
        ("use_internal", BoolArg),
        ("use_mean", BoolArg),
        ("time_it", BoolArg),
    ],
    synopsis="Calculate the distance from distinct structures in a 3D volume to the closest points in another volume.",
)


def volume_distance_multi(
    session,
    sources: List[Model],
    tos: List[Model] | None = None,
    surface_radius: float = 1.0,
    surface_point_tol: int = 2,
    cluster_normal_k: int = 50,
    cluster_region_k: int = 30,
    cluster_curv_threshold: float | None = None,
    cluster_angle_threshold: float | None = None,
    bond_radius: float = 50.0,
    use_surface: bool = True,
    use_internal: bool = False,
    use_mean: bool = True,
    time_it: bool = True,
) -> None:
    """Calculate distances from a segmented volume/surface to the closest other segmented volume/surface and displays them."""
    session.logger.info("Loading volume data...")
    # Check types
    if not all([isinstance(source, (Volume, Surface)) for source in sources]):
        session.logger.error("Source models must be a Volume or Surface.")
        return

    if tos is not None and not all([isinstance(to, (Volume, Surface)) for to in tos]):
        session.logger.error(
            "Target model must be a Volume or Surface or not specified."
        )
        return
    try:
        if time_it:
            import time

            start_time = time.time()
        session.logger.info(
            f"Calculating distances from sources '{[source.name for source in sources]}' to targets '{[to.name for to in tos] if tos else 'self'}' with use_surface={use_surface}, and use_internal={use_internal}."
        )
        if tos is None:
            distances = _calculate_intravolume_distance(
                session,
                sources,
                surface_radius=surface_radius,
                surface_point_tol=surface_point_tol,
                cluster_normal_k=cluster_normal_k,
                cluster_region_k=cluster_region_k,
                cluster_curv_threshold=cluster_curv_threshold,
                cluster_angle_threshold=cluster_angle_threshold,
                bond_radius=bond_radius,
                use_surface=use_surface,
                use_internal=use_internal,
                use_mean=use_mean,
            )
        else:
            distances = _calculate_intervolume_distance(
                session,
                sources,
                tos,
                surface_radius=surface_radius,
                surface_point_tol=surface_point_tol,
                cluster_normal_k=cluster_normal_k,
                cluster_region_k=cluster_region_k,
                cluster_curv_threshold=cluster_curv_threshold,
                cluster_angle_threshold=cluster_angle_threshold,
                bond_radius=bond_radius,
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
    required=[("sources", ModelsArg)],
    keyword=[
        ("tos", ModelsArg),
        ("surface_radius", FloatArg),
        ("surface_point_tol", IntArg),
        ("cluster_normal_k", IntArg),
        ("cluster_region_k", IntArg),
        ("cluster_curv_threshold", FloatArg),
        ("cluster_angle_threshold", FloatArg),
        ("bond_radius", FloatArg),
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
    surface_radius: float = 1.0,
    surface_point_tol: int = 2,
    cluster_normal_k: int = 50,
    cluster_region_k: int = 30,
    cluster_curv_threshold: float | None = None,
    cluster_angle_threshold: float | None = None,
    bond_radius: float = 50.0,
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
            f"Calculating distances from source group '{source.name}' to target group '{to.name if to else 'self'}' with use_surface={use_surface}, and use_internal={use_internal}."
        )
        source_models = [
            m for m in source.child_models() if isinstance(m, (Volume, Surface))
        ]
        if len(source_models) == 0:
            session.logger.info("Couldn't find any volumes inside source group.")
            return
        session.logger.info("Found %d volumes in source group.", len(source_models))
        if to is None:
            distances = _calculate_intravolume_distance(
                session,
                source_models,
                surface_radius=surface_radius,
                surface_point_tol=surface_point_tol,
                cluster_normal_k=cluster_normal_k,
                cluster_region_k=cluster_region_k,
                cluster_curv_threshold=cluster_curv_threshold,
                cluster_angle_threshold=cluster_angle_threshold,
                bond_radius=bond_radius,
                use_internal=use_internal,
                use_mean=use_mean,
            )
        else:
            source_models = [
                m for m in to.child_models() if isinstance(m, (Volume, Surface))
            ]
            if len(source_models) == 0:
                session.logger.info("Couldn't find any volumes inside target group.")
                return
            distances = _calculate_intervolume_distance(
                session,
                source_models,
                source_models,
                surface_radius=surface_radius,
                surface_point_tol=surface_point_tol,
                cluster_normal_k=cluster_normal_k,
                cluster_region_k=cluster_region_k,
                cluster_curv_threshold=cluster_curv_threshold,
                cluster_angle_threshold=cluster_angle_threshold,
                bond_radius=bond_radius,
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
        ("surface_radius", FloatArg),
        ("surface_point_tol", IntArg),
        ("cluster_normal_k", IntArg),
        ("cluster_region_k", IntArg),
        ("cluster_curv_threshold", FloatArg),
        ("cluster_angle_threshold", FloatArg),
        ("bond_radius", FloatArg),
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


def _separate_structures(
    points: np.ndarray,
    normal_k: int = 50,
    region_k: int = 30,
    curv_threshold: float = None,
    angle_threshold: float = None,
) -> Tuple[List[np.array], List[np.ndarray]]:
    """Separate 3D classifications into distinct structures using PCL region growing algorithm and return found cluster ids and points."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(pcd)
    # Estimate normals for the point cloud
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(normal_k))
    pcd.normalize_normals()
    pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamKNN(normal_k))
    points = np.asarray(pcd.points, dtype=np.float32)  # N x 3
    normals = np.asarray(pcd.normals, dtype=np.float32)  # N x 3
    covariances = np.asarray(pcd.covariances, dtype=np.float32)  # N x 3 x 3
    curvatures = np.empty(len(normals), dtype=np.float32)  # N
    # Calculate curvature from eigenvalues
    for i in range(len(covariances)):
        eig_vals = np.linalg.eigvals(covariances[i])
        # Note: based on PCA -> percentage of variance explained by the first component
        # (i.e., how flat the surface is)
        curvatures[i] = eig_vals[np.argmin(eig_vals)] / np.sum(eig_vals)

    # Set default values for thresholds if not provided
    if curv_threshold is None:
        curv_threshold = np.percentile(curvatures, 75)
    if angle_threshold is None:
        angle_threshold = 45.0 * math.pi / 180.0  # 45 degrees in radians

    # Perform region growing clustering
    order = np.argsort(curvatures).tolist()
    clusters = np.full(len(points), -1, dtype=np.int32)  # -1 for noise points
    cur_cluster = 0
    while len(order) > 0:
        seeds = SimpleQueue()
        idx = order.pop(0)
        p_min = points[idx]
        clusters[idx] = cur_cluster
        seeds.put((idx, p_min))
        while not seeds.empty():
            seed_idx, cur_seed = seeds.get()
            _, idx, _ = tree.search_knn_vector_3d(cur_seed.astype(np.float64), region_k)
            for i in idx:
                # If the point isn't in a cluster and its normal is within the angle threshold of the seed, add it to the cluster
                if (
                    i in order
                    and np.arccos(np.abs(np.dot(normals[seed_idx], normals[i])))
                    < angle_threshold
                ):
                    clusters[i] = cur_cluster
                    order.remove(i)
                    # If curvature is below threshold, add to cluster
                    if curvatures[i] < curv_threshold:
                        seeds.put((i, points[i]))
        # No more seeds -> cluster is done
        cur_cluster += 1
    # Clusters is now a 1D array of cluster ids, where -1 is noise points
    cluster_ids = np.unique(clusters)
    cluster_arrays = []
    for cluster_id in cluster_ids:
        if cluster_id == -1:
            # Skip noise points from distance calculations
            continue
        mask = clusters == cluster_id
        cluster_points = points[mask]
        cluster_arrays.append(cluster_points)
    clusters += 1  # 0-index clusters (i.e., outliers are 0)
    return clusters, cluster_arrays  # cluster ids returned for display purposes


def _calculate_surface_points(
    points: np.ndarray, radius: float = 1.0, point_tol: int = 2
) -> np.ndarray:
    """Calculate surface points of a 3D volume as an nx3 numpy array using surface normal estimation."""
    # Get normals with covariance matrix estimation using ball-NN open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius))
    pcd.normalize_normals()
    # Convert to numpy array
    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)
    # Calculate centers of radius neighbourhoods in the +- normal directions
    f_centers = points + normals * radius / 2
    b_centers = points - normals * radius / 2
    # Create KDTree for fast nearest neighbour search
    tree = o3d.geometry.KDTreeFlann(pcd)
    # Count the number of points in each +- normal direction neighbourhood
    f_neighbours = np.array(
        [
            len(tree.search_hybrid_vector_3d(center, radius / 2, point_tol)[1])
            for center in f_centers
        ]
    )
    b_neighbours = np.array(
        [
            len(tree.search_hybrid_vector_3d(center, radius / 2, point_tol)[1])
            for center in b_centers
        ]
    )
    # Points with less than point_tol neighbours in either direction are considered surface points
    f_surfaces = f_centers[np.argwhere(f_neighbours < point_tol)]
    b_surfaces = b_centers[np.argwhere(b_neighbours < point_tol)]
    surface_points = np.unique(np.vstack((f_surfaces, b_surfaces)), axis=0)
    return surface_points


def _calculate_distance(
    point: np.array, surface_points: np.ndarray
) -> Tuple[np.array, float]:
    """Calculate the distance from each point to the closest point in a list of surface points using OcTrees."""
    # Create KDTree for fast nearest neighbour search
    tree = spatial.KDTree(surface_points)
    distance, closest_idx = tree.query(point, workers=-1)
    if distance == np.inf or closest_idx == tree.n:  # No neighbours found
        return point, 0
    closest_point = tree.data[closest_idx]
    return np.array(closest_point), float(distance)


def _calculate_distances(
    points: np.ndarray, surface_points: np.ndarray
) -> Tuple[np.ndarray, np.array]:
    """Calculate the distance from each point to the closest point in a list of surface points using OcTrees."""
    # Create KDTree for fast nearest neighbour search
    tree = spatial.KDTree(surface_points)
    distances, closest_idxs = tree.query(points, workers=-1)
    closest_points = tree.data[closest_idxs]
    return np.array(closest_points), np.array(distances)


def _calculate_distance_internal(
    point: np.array, surface_points: np.ndarray, max_distance: float = np.inf
) -> Tuple[np.array, float]:
    """Calculate the distance from each point to the closest point in a list of surface points using ChimeraX find_closest_points method."""
    from chimerax.geometry import find_closest_points

    _, _, closest_idxs = find_closest_points(
        np.expand_dims(point, axis=0), surface_points, max_distance=max_distance
    )
    closest_point = surface_points[closest_idxs[0]]
    distance = np.linalg.norm(point - closest_point, axis=1)
    return np.array(closest_point), float(distance)


def _calculate_distances_internal(
    points: np.ndarray, surface_points: np.ndarray, max_distance: float = np.inf
) -> Tuple[np.ndarray, np.array]:
    """Calculate the distance from each point to the closest point in a list of surface points using ChimeraX find_closest_points method."""
    from chimerax.geometry import find_closest_points

    _, _, closest_idxs = find_closest_points(
        points, surface_points, max_distance=max_distance
    )
    closest_points = surface_points[closest_idxs]
    distances = np.linalg.norm(points - closest_points, axis=1)
    return np.array(closest_points), np.array(distances)


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
            if closest_point is None or distance is None:
                session.logger.warning(
                    f"Could not calculate distance for the cluster {i}'."
                )
                continue
            else:
                source_points[i] = point
                closest_points[i] = closest_point
                distances[i] = distance
        else:
            if use_internal:
                # Use ChimeraX find_closest_points method
                closest_points, distances = _calculate_distances_internal(
                    s_c, target_surface_points, max_distance=np.inf
                )
            else:
                # Use octree method
                closest_points, distances = _calculate_distances(
                    s_c, target_surface_points
                )
            source_points = s_c
    # Remove zero-distance or inf-distance clusters
    source_points = source_points[~np.logical_or(distances == 0, distances == np.inf)]
    closest_points = closest_points[~np.logical_or(distances == 0, distances == np.inf)]
    distances = distances[~np.logical_or(distances == 0, distances == np.inf)]
    return source_points, closest_points, distances


def _display_distance(
    session,
    source_points: np.ndarray,
    closest_points: np.ndarray,
    distances: np.array,
    radius: float = 100,
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
    bonds = []
    for source_marker, surface_marker, distance in zip(
        source_markers, surface_markers, distances
    ):
        link = create_link(source_marker, surface_marker, rgba=color, radius=radius)
        bonds.append(link)
        b = Objects(bonds=Bonds([link]))
        label(
            session,
            objects=b,
            object_type="bonds",
            text=str(distance),
            height=500,
            color=Color(color),
        )

    # Plot distances on a histogram
    from .plot import VolumeDistancePlot

    plot = VolumeDistancePlot(session, Bonds(bonds))


# ============================================================================
# Internal functions for running commands
# ============================================================================


def _calculate_intravolume_distance(
    session,
    sources: List[Model],
    surface_radius: float = 1.0,
    surface_point_tol: int = 2,
    cluster_normal_k: int = 50,
    cluster_region_k: int = 30,
    cluster_curv_threshold: float | None = None,
    cluster_angle_threshold: float | None = None,
    bond_radius: float = 50.0,
    use_surface: bool = True,
    use_internal: bool = False,
    use_mean: bool = True,
) -> np.array:
    """Calculate distances from a segmented volume/surface to the closest points in itself and displays them."""
    if use_surface:
        surfaces = []
        for model in sources:
            if isinstance(model, Surface):
                surfaces.append(model)
            elif isinstance(model, Volume) and len(model.surfaces) > 0:
                surfaces.append(model.surfaces[0])
            else:
                session.logger.warning(
                    f"Model {model.name} is not a Surface or Volume with surfaces, skipping."
                )
        surface_points = [
            np.array(surf.vertices).astype(np.float32) for surf in surfaces
        ]
    else:
        if not all([isinstance(s, Volume) for s in sources]):
            session.logger.error("All source models must be Volumes.")
            return np.array([])
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
                _calculate_surface_points(
                    source_idx, radius=surface_radius, point_tol=surface_point_tol
                ).astype(np.float32)
            )
    # Cluster points into distinct structures if single volume
    if len(surface_points) == 1:
        from chimerax.core.colors import BuiltinColormaps

        session.logger.info(
            "Only one source volume provided, clustering surface points into distinct structures."
        )
        surface = (
            sources[0] if isinstance(sources[0], Surface) else sources[0].surfaces[0]
        )
        surface_points = surface_points[0]
        surface_clusters, surface_points = _separate_structures(
            surface_points,
            normal_k=cluster_normal_k,
            region_k=cluster_region_k,
            curv_threshold=cluster_curv_threshold,
            angle_threshold=cluster_angle_threshold,
        )
        # Map clusters to distinct colors
        mask_surface = Surface(f"{surface.name}_clusters", session)
        mask_surface.set_geometry(surface.vertices, surface.normals, surface.triangles)
        palette = BuiltinColormaps["rainbow"].rescale_range(
            np.min(surface_clusters), np.max(surface_clusters)
        )
        mask_surface.vertex_colors = palette.interpolated_rgba8(surface_clusters)
        session.models.add([mask_surface])
        session.logger.info(
            f"Separated {len(surface_points)} distinct structures from the source volume."
        )
    # Calculate distances between clusters
    source_points, closest_points, distances = _calculate_distances_between_clusters(
        session,
        surface_points,
        [np.copy(surf) for surf in surface_points],
        use_internal=use_internal,
        use_mean=use_mean,
    )
    # Display distance
    _display_distance(
        session, source_points, closest_points, distances, radius=bond_radius
    )
    return distances


def _calculate_intervolume_distance(
    session,
    sources: List[Model],
    targets: List[Model],
    surface_radius: float = 1.0,
    surface_point_tol: int = 2,
    cluster_normal_k: int = 50,
    cluster_region_k: int = 30,
    cluster_curv_threshold: float | None = None,
    cluster_angle_threshold: float | None = None,
    bond_radius: float = 50.0,
    use_surface: bool = True,
    use_internal: bool = False,
    use_mean: bool = True,
) -> np.array:
    """Calculate distances from a segmented volume/surface to the closest other segmented surface and displays them."""
    if use_surface:
        source_surfaces = []
        target_surfaces = []
        for model in sources:
            if isinstance(model, Surface):
                source_surfaces.append(model)
            elif isinstance(model, Volume) and len(model.surfaces) > 0:
                source_surfaces.append(model.surfaces[0])
            else:
                session.logger.warning(
                    f"Source model {model.name} is not a Surface or Volume with surfaces, skipping."
                )
        for model in targets:
            if isinstance(model, Surface):
                target_surfaces.append(model)
            elif isinstance(model, Volume) and len(model.surfaces) > 0:
                target_surfaces.append(model.surfaces[0])
            else:
                session.logger.warning(
                    f"Target model {model.name} is not a Surface or Volume with surfaces, skipping."
                )
        source_surface_points = [
            np.array(surf.vertices).astype(np.float32) for surf in source_surfaces
        ]
        target_surface_points = [
            np.array(surf.vertices).astype(np.float32) for surf in target_surfaces
        ]
    else:
        if not all([isinstance(s, Volume) for s in sources]):
            session.logger.error("All source models must be Volumes.")
            return np.array([])
        if not all([isinstance(t, Volume) for t in targets]):
            session.logger.error("All source models must be Volumes.")
            return np.array([])
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
                _calculate_surface_points(
                    source_idx, radius=surface_radius, point_tol=surface_point_tol
                ).astype(np.float32)
            )
        for volume in targets:
            data_matrix = volume.full_matrix()
            target_idx = np.argwhere(data_matrix > 0).astype(np.uint16)
            if target_idx.size == 0:
                raise ValueError(
                    f"Target volume {volume.name} does not contain any non-zero data."
                )
            target_surface_points.append(
                _calculate_surface_points(
                    target_idx, radius=surface_radius, point_tol=surface_point_tol
                ).astype(np.float32)
            )
    # Cluster points into distinct structures if single volume
    if len(source_surface_points) == 1:
        from chimerax.core.colors import BuiltinColormaps

        session.logger.info(
            "Only one source volume provided, clustering surface points into distinct structures."
        )
        source_surface = (
            sources[0] if isinstance(sources[0], Surface) else sources[0].surfaces[0]
        )
        source_surface_points = source_surface_points[0]
        source_surface_clusters, source_surface_points = _separate_structures(
            source_surface_points,
            normal_k=cluster_normal_k,
            region_k=cluster_region_k,
            curv_threshold=cluster_curv_threshold,
            angle_threshold=cluster_angle_threshold,
        )
        # Map clusters to distinct colors
        mask_surface = Surface(f"{source_surface.name}_clusters", session)
        mask_surface.set_geometry(
            source_surface.vertices, source_surface.normals, source_surface.triangles
        )
        palette = BuiltinColormaps["rainbow"].rescale_range(
            np.min(source_surface_clusters), np.max(source_surface_clusters)
        )
        mask_surface.vertex_colors = palette.interpolated_rgba8(source_surface_clusters)
        session.models.add([mask_surface])
        session.logger.info(
            f"Separated {len(source_surface_points)} distinct structures from the source volume."
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
    _display_distance(session, source_points, closest_points, distances, bond_radius)
    return distances
