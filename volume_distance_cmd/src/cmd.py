from .octree import Point, OcTree

import numpy as np
from scipy import spatial, cluster

from chimerax.core.commands import CmdDesc, register, VolumeArg
               
# ============================================================================
# Functions and descriptions for registering using ChimeraX bundle API
# ============================================================================


def volume_distance(
    session, source: Volume, to: Volume, internal: bool = False
) -> None:
    """Calculate distances from a segmented volume to the closest other segmented volume and displays them."""
    source_data, target_data = source.data, to.data
    _calculate_volume_distance(session, source_data, target_data, internal)
    session.logger.info(
        f"Calculated distances from volume '{source.name}' to volume '{to.name}'."
    )


volume_distance_desc = CmdDesc(
        required=[("source", VolumeArg)],
        keywords=[("to", VolumeArg), ("internal", bool)],
        required_arguments=["to"],
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
        return float('inf')
    # Get the top-left and bottom-right corners of the node
    top_left, bottom_right = node.top_left, node.bottom_right
    # Calculate distances
    dx = max(bottom_right.x - point.x, point.x - top_left.x, 0)
    dy = max(bottom_right.y - point.y, point.y - top_left.y, 0)
    dz = max(bottom_right.z - point.z, point.z - top_left.z, 0)
    # Return the Euclidean distance from the point to the closest point on the node's surface
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
    
def _find_closest_point_octree(point: Point, octree: OcTree) -> Point:
    """Find the closest point in the octree to the given point."""
    queue = PriorityQueue()
    queue.put((0, octree))  # Start with the root node and distance 0
    while not queue.empty():
        current_distance, current_node = queue.get()
        # Check for internal node and add children to the queue
        if current_node.point is None:
            for child in current_node.children:
                if child.point is None:
                    # Calculate the distance from the point to the child node
                    distance_to_child = calculate_distance_to_node(point, child)
                    queue.put((distance_to_child, child))
                elif child.point == Point(-1, -1, -1):
                    # If the child is an empty node, we can skip it
                    continue
                else:
                    # If the child is a leaf node with a point, calculate the distance
                    distance = calculate_distance(point, child.point)
                    queue.put((distance, child))
        # Return distance if the current node is a leaf node with a point
        elif current_node.point != Point(-1, -1, -1):
            return current_node.point
        else:
            # No empty nodes should be added to the queue
            sesssion.logger.error(
                f"Unexpected state: current node is neither internal nor leaf with a point."
            )
            return

#### Define ChimeraX commands for calculating distances ####


def _separate_structures(volume: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """Separate 3D classifications into distinct structures using ward clustering."""
    points = np.argwhere(volume > 0)
    Z = cluster.hierarchy.ward(points)
    cluster_idxs = cluster.hierarchy.fcluster(Z, radius, criterion='distance')
    clusters = np.unique(cluster_idxs)
    separated_volumes = np.array([], dtype=np.uint8)
    for cluster_id in clusters:
        mask = cluster_idxs == cluster_id
        cluster_points = points[mask]
        if len(cluster_points) > 0:
            separated_volume = np.zeros_like(volume, dtype=np.uint8)
            separated_volume[tuple(cluster_points.T)] = 1
            separated_volumes.append(separated_volume)
    return separated_volumes

def _calculate_surface_points(volume: np.ndarray, radius: float = 1.0, point_tol: int = 1) -> np.ndarray:
    """Calculate surface points of a 3D volume as an nx3 numpy array using surface normal estimation."""
    # Get points from volume
    points = np.argwhere(volume > 0)
    tree = spatial.KDTree(points)
    surface_points = []
    for point in points:
        # Find local neighbourhood around query point
        idx = tree.query_ball_point(point, radius, eps=0, p=2)
        nbrhood = points[idx]
        # Estimate normal direction at query point using SVD
        nbrhood_centered = nbrhood - np.mean(nbrhood, axis=0)
        nbrhood_cov = np.cov(nbrhood_centered)
        U, S, _ = np.linalg.svd(nbrhood_cov)
        # Use smallest eigenvector as normal direction
        normal = U[:, np.argmin(S)]
        # Search for points in circular patches in the normal direction
        centers = [point + normal * radius / 2, point - normal * radius / 2]
        for center in centers:
            nbrhood_idx = tree.query_ball_point(center, radius / 2, eps=0, p=2)
            if len(nbrhood_idx) < point_tol:
                surface_points.append(center)
    return np.unique(surface_points, axis=0)


def _calculate_distance(
    points: np.ndarray, surface_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the distance from each point to the closest point in a list of surface points using OcTrees."""
    top_left, bottom_right = Point(0, 0, 0), Point(*surface_points.shape)
    octree = OcTree(
        point=None, top_left=top_left, bottom_right=bottom_right
    )  # Create an empty octree
    closest_points = np.array([], dtype=np.float32)
    distances = np.array([], dtype=np.float32)
    for point in points:
        closest_point = _find_closest_point_octree(Point(*point), octree)
        if closest_point is None:
            continue
        distance = _calculate_distance_between_points(Point(*point), closest_point)
        closest_points.append(np.array([closest_point.x, closest_point.y, closest_point.z]))
        distances.append(distance)
    return closest_points, distances


def _calculate_distance_internal(
    points: np.ndarray, surface_points: np.ndarray, max_distance: float = 1000.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the distance from each point to the closest point in a list of surface points using ChimeraX find_closest_points method."""
    _, _, closest_points = find_closest_points(points, surface_points, max_distance=max_distance)
    distances = np.linalg.norm(points - closest_points, axis=1)
    return closest_points, distances


def _display_distance(session, source_points: np.ndarray, closest_points: np.ndarray, distances: np.array) -> None:
    """Display the distance between two points in ChimeraX."""
    pass

def _calculate_volume_distance(
    session, source_volume: np.ndarray, target_volume: np.ndarray, internal: bool = False, time_it: bool = True
) -> None:
    """Calculate distances from a segmented volume to the closest other segmented volume and displays them."""
    if time_it:
        import time
        start_time = time.time()
        
    source_structures = _separate_structures(source_volume)
    source_coords = np.array([np.argwhere(structure > 0) for structure in source_structures])
    source_points = np.mean(source_coords, axis=0) # n x 3 where n is the number of structures
    surface_points = _calculate_surface_points(target_volume)
    if internal:
        # Use ChimeraX find_closest_points method
        closest_points, distances = _calculate_distance_internal(source_points, surface_points, max_distance=np.max(source_volume.shape))
    else:
        # Use octree method
        closest_points, distances = _calculate_distance(source_points, surface_points)
    # closest_points is n x 3 and distances is n x 1
    _display_distance(session, source_points, closest_points, distances)
    
    if time_it:
        end_time = time.time()
        session.logger.info(
            f"Calculated distances in {end_time - start_time:.2f} seconds."
        )