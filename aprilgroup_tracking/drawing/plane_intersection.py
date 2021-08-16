"""
Module to measure the distance between a point and a plane.
"""
import numpy as np
from logging import Logger
import math 


class PlaneIntersection:
    """
    """

    def __init__(self, logger: Logger, world_coords) -> None:
        self.logger = logger
        self.world_coords: np.ndarray = world_coords
        self.norm_vector: np.ndarray = self._get_norm_vector(world_coords)
        self.d: float

    def _get_norm_vector(self, world_coords):
        """Obtain the perpendicular (normal) vector of the plane
        with three points that lie on the plane."""

        point0_point1 = np.subtract(self.world_coords[1], self.world_coords[0])
        point0_point2 = np.subtract(self.world_coords[2], self.world_coords[0])
        norm_vector = np.cross(point0_point1, point0_point2)

        return norm_vector

    # Get D if there is ones
    def get_d(self, plane_point, test_point):
        """Obtain d.
        """

        (norm_a, norm_b, norm_c) = self.norm_vector
        (x, y, z) = plane_point
        d = norm_a*x + norm_b*y + norm_c*z

        plane_eq_test = self.test_plane_eq(test_point, norm_a, norm_b, norm_c, d)

        #TODO: If plane_eq_test == 0 for all 3 test points, save d
        self.d = -d

    def test_plane_eq(self, plane_point, norm_a, norm_b, norm_c, d):
        """Equation should return 0 if accurate.

        Plane equation = A*(x - x0) + B*(y - y0) + C*(z - z0) + D
        """

        (x, y, z) = plane_point
        return norm_a*x + norm_b*y + norm_c*z + d

    def shortest_distance(self, x1, y1, z1):
        """Obtain the shortest distance between the plane
        and a point.
        """

        (norm_a, norm_b, norm_c) = self.norm_vector
        distance = abs((norm_a * x1 + norm_b * y1 + norm_c * z1 + self.d))
        norm = (math.sqrt(norm_a * norm_a + norm_b * norm_b + norm_c * norm_c))
        return distance/norm

    def check_distance(self, radius, distance):
        """Checks if the distance is within the radius of the point.
        """

        return distance > -radius and distance < radius
