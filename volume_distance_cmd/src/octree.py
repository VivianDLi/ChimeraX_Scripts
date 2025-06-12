from enum import IntEnum
from dataclasses import dataclass, field
from typing import List

#### Logging Setup ####
import logging

logger = logging.getLogger(__name__)

#### Define the Octree structure and related classes for efficient distance calculation ####


class Octant(IntEnum):
    TOP_LEFT_FRONT = 0
    TOP_RIGHT_FRONT = 1
    BOTTOM_LEFT_FRONT = 2
    BOTTOM_RIGHT_FRONT = 3
    TOP_LEFT_BACK = 4
    TOP_RIGHT_BACK = 5
    BOTTOM_LEFT_BACK = 6
    BOTTOM_RIGHT_BACK = 7


@dataclass
class Point:
    x: float
    y: float
    z: float


@dataclass
class OcTree:
    point: Point = field(
        default_factory=Point(-1, -1, -1)
    )  # if None, this is a node, if (-1, -1, -1), node is empty
    top_left: Point = None  # coordinates of the top-left corner of the octree node
    bottom_right: Point = (
        None  # coordinates of the bottom-right corner of the octree node
    )
    children: List["OcTree"] = field(init=False)

    def __post_init__(self):
        # Check bounds
        if (
            self.top_left.x >= self.bottom_right.x
            or self.top_left.y >= self.bottom_right.y
            or self.top_left.z >= self.bottom_right.z
        ):
            raise ValueError("Invalid bounds for octree node.")
        # Initialize children as empty nodes
        self.children = [OcTree() for _ in range(8)]

    def insert(self, point: Point):
        """Insert a point into the octree."""
        # Check if the point already exists in the octree
        if self.find(point):
            logger.warning(f"Point {point} already exists in the octree.")
            return
        # Check if the point is within the bounds of this node
        if not self.is_within_bounds(point):
            logger.warning(f"Point {point} is out of bounds for this octree node.")
            return
        ## Binary search to insert the point
        octant = self.get_octant(point)
        # Check for internal node and recursively insert
        if self.children[octant].point is None:
            self.children[octant].insert(point)
            return
        # Check for empty node
        elif self.children[octant].point == Point(-1, -1, -1):
            self.children[octant] = OcTree(
                point=point, top_left=None, bottom_right=None
            )
        # Subdivide node if it already has a point
        else:
            existing_point = self.children[octant].point
            # Create new node for the existing point
            self.children[octant] = self.create_octree(octant)
            # Insert both points into the new node
            self.children[octant].insert(existing_point)
            self.children[octant].insert(point)

    def find(self, point: Point) -> bool:
        """Check if a point exists in the octree."""
        if not self.is_within_bounds(point):
            return False
        ## Binary search to find the point
        octant = self.get_octant(point)
        # Check for internal node and recursively search
        if self.children[octant].point is None:
            return self.children[octant].find(point)
        # Check for empty node
        elif self.children[octant].point == Point(-1, -1, -1):
            return False
        # Check if the point matches the stored point
        elif self.children[octant].point == point:
            return True
        # If the point does not match, return False
        else:
            return False

    def is_within_bounds(self, point: Point) -> bool:
        """Check if a point is within the bounds of this octree node."""
        return (
            self.top_left.x <= point.x <= self.bottom_right.x
            and self.top_left.y <= point.y <= self.bottom_right.y
            and self.top_left.z <= point.z <= self.bottom_right.z
        )

    def get_octant(self, point: Point) -> Octant:
        """Get the octant of a point within this octree node."""
        mid_x = (self.top_left.x + self.bottom_right.x) / 2
        mid_y = (self.top_left.y + self.bottom_right.y) / 2
        mid_z = (self.top_left.z + self.bottom_right.z) / 2

        if point.x < mid_x:
            if point.y < mid_y:
                if point.z < mid_z:
                    return Octant.TOP_LEFT_FRONT
                else:
                    return Octant.TOP_LEFT_BACK
            else:
                if point.z < mid_z:
                    return Octant.BOTTOM_LEFT_FRONT
                else:
                    return Octant.BOTTOM_LEFT_BACK
        else:
            if point.y < mid_y:
                if point.z < mid_z:
                    return Octant.TOP_RIGHT_FRONT
                else:
                    return Octant.TOP_RIGHT_BACK
            else:
                if point.z < mid_z:
                    return Octant.BOTTOM_RIGHT_FRONT
                else:
                    return Octant.BOTTOM_RIGHT_BACK

    def create_octree(self, octant: Octant) -> "OcTree":
        """Create a new octree node for the specified octant."""
        mid_x = (self.top_left.x + self.bottom_right.x) / 2
        mid_y = (self.top_left.y + self.bottom_right.y) / 2
        mid_z = (self.top_left.z + self.bottom_right.z) / 2

        match octant:
            case Octant.TOP_LEFT_FRONT:
                return OcTree(
                    top_left=self.top_left,
                    bottom_right=Point(mid_x, mid_y, mid_z),
                )
            case Octant.TOP_RIGHT_FRONT:
                return OcTree(
                    top_left=Point(mid_x + 1, self.top_left.y, self.top_left.z),
                    bottom_right=Point(self.bottom_right.x, mid_y, mid_z),
                )
            case Octant.BOTTOM_LEFT_FRONT:
                return OcTree(
                    top_left=Point(self.top_left.x, mid_y + 1, self.top_left.z),
                    bottom_right=Point(mid_x, self.bottom_right.y, mid_z),
                )
            case Octant.BOTTOM_RIGHT_FRONT:
                return OcTree(
                    top_left=Point(mid_x + 1, mid_y + 1, self.top_left.z),
                    bottom_right=Point(self.bottom_right.x, self.bottom_right.y, mid_z),
                )
            case Octant.TOP_LEFT_BACK:
                return OcTree(
                    top_left=Point(self.top_left.x, self.top_left.y, mid_z + 1),
                    bottom_right=Point(mid_x, mid_y, self.bottom_right.z),
                )
            case Octant.TOP_RIGHT_BACK:
                return OcTree(
                    top_left=Point(mid_x + 1, self.top_left.y, mid_z + 1),
                    bottom_right=Point(self.bottom_right.x, mid_y, self.bottom_right.z),
                )
            case Octant.BOTTOM_LEFT_BACK:
                return OcTree(
                    top_left=Point(self.top_left.x, mid_y + 1, mid_z + 1),
                    bottom_right=Point(mid_x, self.bottom_right.y, self.bottom_right.z),
                )
            case Octant.BOTTOM_RIGHT_BACK:
                return OcTree(
                    top_left=Point(mid_x + 1, mid_y + 1, mid_z + 1),
                    bottom_right=self.bottom_right,
                )
