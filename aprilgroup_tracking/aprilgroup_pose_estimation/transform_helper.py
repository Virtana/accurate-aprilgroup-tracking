import numpy as np
import cv2
from typing import List, Dict, Tuple
from numpy.testing import assert_array_almost_equal
# import the math module
from math import sqrt, sin, cos, atan2


class TransformHelper(object):
    """Helper functions relating to object points, appending values,
    transformation, extrinsic matrices and pose velocity and acceleration.

    Attributes:
        logger: Used to create class specific logs for info and debugging.
        mtx: Camera Matrix.
        dist: Camera Distortion Coefficients.
    """

    def __init__(self, logger, mtx, dist):
        self.logger = logger
        self.mtx: np.ndarray = mtx
        self.dist: np.ndarray = dist

    def add_values_in_dict(
        self,
        sample_dict: Dict,
        key: int,
        list_of_values: List[object]
    ) -> Dict:
        """Appends multiple values to a key in the given dictionary
        """

        if key not in sample_dict:
            sample_dict[key] = list()
        sample_dict[key].extend(list_of_values)

        return sample_dict

    def get_initial_pts(self, markersize: float) -> np.ndarray:
        """Obtains the initial 3D points in space of AprilTags.

        Args:
        markersize:
            Size of the apriltag.

        Returns:
        Initial 3D points of AprilTags.
        """

        # AprilTag Radius
        mhalf = markersize / 2.0

        # AprilTag Initial 3D points in space
        ob_pt1 = [-mhalf, -mhalf, 0.0]  # Lower left in marker world
        ob_pt2 = [-mhalf, mhalf, 0.0]   # Upper left in marker world
        ob_pt3 = [mhalf,  mhalf, 0.0]   # Upper right in marker world
        ob_pt4 = [mhalf,  -mhalf, 0.0]  # Lower right in marker world
        ob_pts = ob_pt1 + ob_pt2 + ob_pt3 + ob_pt4
        object_pts = np.array(ob_pts).reshape(4, 3)

        return object_pts

    def transform_marker_corners(
        self,
        object_pts: np.ndarray,
        transformation: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """Rotates and Translates the apriltag by the rotation
        and translation vectors given.

        Args:
        object_pts:
            Initial 3D points in space
        transformation:
            Rotation and Translation vector, respectively, of the apriltag.

        Returns:
        3D points of Apriltag after rotation and translation has been applied,
        to be supplied to solvePnP() to obtain the pose of the apriltag.
        """

        # Convert rotation vector to rotation matrix (markerworld -> cam-world)
        mrv = cv2.Rodrigues(transformation[0])[0]

        # Apply rotation to 3D points in space
        app_rot_mat = object_pts @ mrv.T

        # Apply Translation to 3D points in space
        tran_vec = transformation[1].reshape(-1, 3)
        tran_mat = app_rot_mat + tran_vec

        return tran_mat

    def get_reprojection_error(
        self,
        obj_points,
        img_points,
        transformation
    ) -> float:
        """Calculates the reprojection error betwen
        the object and image points given the object pose.
        """

        # Obtains the project image points for the object
        # points and transformation given.
        try:
            project_points, _ = cv2.projectPoints(
                obj_points,
                transformation[0],
                transformation[1],
                self.mtx,
                self.dist)
        except(RuntimeError, TypeError) as error:
            raise error

        # Reshape to fit numpy functions
        project_points = project_points.reshape(-1, 2)

        # Calculates the average reprojection error
        reprojection_error_avg = sum(
            [np.linalg.norm(img_points[i] - project_points[i])
                for i in range(len(project_points))]) / len(project_points)

        return reprojection_error_avg

    def get_extrinsic_matrix(self, rmat, tvec):
        """
        Returns the extrinsic matrix containing the rotational matrix
        and translation vector in the form:
        [R1,1 R1,2 R1,3 T1]
        [R2,1 R2,2 R2,3 T2]
        [R3,1 R3,2 R3,3 T3]
        [0     0    0    1].
        """

        extrinsic_mat = np.vstack(
            (np.hstack(
                (rmat, tvec)),
                np.array([0, 0, 0, 1]))
            )
        self.logger.info(
            "\n Rmat: {} \n Tvec: {}".format(rmat, tvec))
        self.logger.info(
            "\n Extrinsic Matrix OBTAINED: {}".format(extrinsic_mat))

        return extrinsic_mat

    def get_rmat_tvec(self, extrinsic_mat):
        """ Extracts the rotation matrix
        and translation vector from an extrinsic matrix.
        """
        rel_rot_mat = extrinsic_mat[0:3, 0:3]
        rel_tvec = np.array(
            extrinsic_mat[0:3, 3], dtype=np.float32).reshape(3, -1)

        return rel_rot_mat, rel_tvec

    def get_relative_trans(self, rot_mat, t1, t0):
        """Obtains the translational velocity between two frames.

        Args:
        prev_rot:
            Previous frame rotation.
        t0:
            First (or previous) Translation Vector.
        t1:
            Second (or current) Translation Vector.

        Returns:
        The translational velocity between two frames, when t = 0.
        """

        # Relative translation between frames
        try:
            tran_vel = (rot_mat.T) @ (t0 - t1)
        except(RuntimeError, TypeError) as error:
            raise error

        return tran_vel

    def get_relative_rot(self, r0, r1):
        """Obtains the rotation matrix between two frames.

        Args:
        r0:
            First (or previous) Rotation.
        r1:
            Second (or current) Rotation.

        Returns:
        The relative rotation matrix between two frames,
        also know as the rotational velocity in this case as t = 0.
        """

        try:
            r0_to_r1 = r1.T @ r0
        except(RuntimeError, TypeError) as error:
            raise error

        return r0_to_r1

    @staticmethod
    def euler_angles_to_rotation_matrix(theta):
        """Calculates Rotation Matrix given euler angles."""

        r_x = np.array([[1,                 0,                  0],
                        [0,         cos(theta[0]), -sin(theta[0])],
                        [0,         sin(theta[0]),  cos(theta[0])]
                        ])

        r_y = np.array([[cos(theta[1]),    0,      sin(theta[1])],
                        [0,                1,      0],
                        [-sin(theta[1]),   0,      cos(theta[1])]
                        ])

        r_z = np.array([[cos(theta[2]),    -sin(theta[2]),    0],
                        [sin(theta[2]),    cos(theta[2]),     0],
                        [0,                     0,            1]
                        ])

        rmat = np.dot(r_z, np.dot(r_y, r_x))
        return rmat

    @staticmethod
    def rotation_matrix_to_euler_angles(rmat):
        """Calculates rotation matrix to euler angles.

        The result is the same as MATLAB except the order
        of the euler angles ( x and z are swapped ).
        """
        sy = sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = atan2(rmat[2, 1], rmat[2, 2])
            y = atan2(-rmat[2, 0], sy)
            z = atan2(rmat[1, 0], rmat[0, 0])
        else:
            x = atan2(-rmat[1, 2], rmat[1, 1])
            y = atan2(-rmat[2, 0], sy)
            z = 0

        return np.array([x, y, z])
