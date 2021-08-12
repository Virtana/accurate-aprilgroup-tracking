"""
Transform calculations and helper functions for
detecting and estimation poses of the DodecaPen.
"""

# import the math module
from math import sqrt, sin, cos, atan2
from typing import List, Dict, Tuple
import numpy as np
import cv2 as cv
from numpy.testing import assert_array_almost_equal


class TransformHelper():
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

    @staticmethod
    def add_values_in_dict(sample_dict: Dict, key: int, list_of_values: List[object]) -> Dict:
        """Appends multiple values to a key in the given dictionary
        """

        if key not in sample_dict:
            sample_dict[key] = list()
        sample_dict[key].extend(list_of_values)

        return sample_dict

    @staticmethod
    def get_initial_pts(tagsize: float) -> np.ndarray:
        """Obtains the initial 3D points in space of AprilTags.

        Args:
        tagsize:
            Size of the apriltag.

        Returns:
        Initial 3D points of AprilTags.
        """

        # AprilTag Radius
        tag_radius = tagsize / 2.0

        # AprilTag Initial 3D points in space
        ob_pt1 = [-tag_radius, -tag_radius, 0.0]  # Lower left in marker world
        ob_pt2 = [-tag_radius, tag_radius, 0.0]   # Upper left in marker world
        ob_pt3 = [tag_radius,  tag_radius, 0.0]   # Upper right in marker world
        ob_pt4 = [tag_radius,  -tag_radius, 0.0]  # Lower right in marker world
        ob_pts = ob_pt1 + ob_pt2 + ob_pt3 + ob_pt4
        object_pts = np.array(ob_pts).reshape(4, 3)

        return object_pts

    @staticmethod
    def transform_marker_corners(object_pts: np.ndarray,
                                 transformation: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
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

        if transformation[0].size == 0 or transformation[1].size == 0:
            raise ValueError('The transform rotation or translation: {} entered is empty'.format(
                transformation))

        # Convert rotation vector to rotation matrix (markerworld -> cam-world)
        rmat = cv.Rodrigues(transformation[0])[0]

        # Apply rotation to 3D points in space
        apply_rotation_mat = object_pts @ rmat.T

        # Apply Translation to 3D points in space
        tran_vec = transformation[1].reshape(-1, 3)
        apply_tran_mat = apply_rotation_mat + tran_vec

        return apply_tran_mat

    def get_reprojection_error(self, obj_points: np.ndarray, img_points: np.ndarray,
                               transformation: Tuple[np.ndarray, np.ndarray]) -> float:
        """Calculates the reprojection error betwen
        the object and image points given the object pose.
        """

        # Obtains the project image points for the object
        # points and transformation given.
        project_points, _ = cv.projectPoints(
            obj_points,
            transformation[0],
            transformation[1],
            self.mtx,
            self.dist)

        # Reshape to fit numpy functions
        project_points = project_points.reshape(-1, 2)

        # Calculates the average reprojection error
        reprojection_error_avg = sum(
            [np.linalg.norm(img_points[i] - project_points[i])
                for i in range(len(project_points))]) / len(project_points)

        return reprojection_error_avg

    def get_extrinsic_matrix(self, rmat: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """
        Returns the extrinsic matrix containing the rotational matrix
        and translation vector in the form:
        [R1,1 R1,2 R1,3 T1]
        [R2,1 R2,2 R2,3 T2]
        [R3,1 R3,2 R3,3 T3]
        [0     0    0    1].
        """

        try:
            extrinsic_mat = np.vstack(
                (np.hstack(
                    (rmat, tvec)),
                    np.array([0, 0, 0, 1]))
                )
        except ValueError as no_transform_err:
            raise ValueError('The rotation matrix: {} or translation vector: {} \
                entered are not in the right format (3x3 matrix and 3x1 vector) \
                    or are zero.'.format(rmat, tvec)) from no_transform_err
        self.logger.info(
            "\n Rmat: {} \n Tvec: {}".format(rmat, tvec))
        self.logger.info(
            "\n Extrinsic Matrix OBTAINED: {}".format(extrinsic_mat))

        return extrinsic_mat

    @staticmethod
    def get_rmat_tvec(extrinsic_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Extracts the rotation matrix
        and translation vector from an extrinsic matrix.
        """

        try:
            rot_mat = extrinsic_mat[0:3, 0:3]
            tvec = np.array(
                extrinsic_mat[0:3, 3], dtype=np.float32).reshape(3, -1)
        except ValueError as ext_mat_err:
            raise ValueError('The extrinsic matrix entered: {} \
                is not a 4x4 matrix or is zero.'.format(extrinsic_mat)) from ext_mat_err

        return rot_mat, tvec

    @staticmethod
    def get_relative_trans(rot_mat: np.ndarray, tvec1: np.ndarray, tvec0: np.ndarray) -> np.ndarray:
        """Obtains the translational velocity between two frame.

        Args:
        prev_rot:
            Current frame rotation.
        t0:
            First (or previous) Translation Vector.
        t1:
            Second (or current) Translation Vector.

        Returns:
        The translational velocity between two frames, when t = 0.
        """

        # Relative translation between frames
        try:
            rel_tran = (rot_mat.T) @ (tvec0 - tvec1)
        except ValueError as tvec_err:
            raise ValueError('The vectors entered, tvec0: {} and tvec1: {} \
                are either not the same size or zero.'.format(tvec0, tvec1)) from tvec_err

        return rel_tran

    @staticmethod
    def get_relative_rot(rmat0: np.ndarray, rmat1: np.ndarray) -> np.ndarray:
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
            r1_to_r0 = rmat1.T @ rmat0
        except ValueError as rmat_err:
            raise ValueError('The matrices entered, r0: {} and r1: {} \
                are either not the same size or zero.'.format(rmat0, rmat1)) from rmat_err

        return r1_to_r0

    @staticmethod
    def euler_angles_to_rotation_matrix(theta) -> np.ndarray:
        """Calculates Rotation Matrix given euler angles."""

        print("theta", type(theta))

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
    def rotation_matrix_to_euler_angles(rmat: np.ndarray) -> np.ndarray:
        """Calculates rotation matrix to euler angles.

        The result is the same as MATLAB except the order
        of the euler angles ( x and z are swapped ).
        """

        s_y = sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])

        singular = s_y < 1e-6

        if not singular:
            x_val = atan2(rmat[2, 1], rmat[2, 2])
            y_val = atan2(-rmat[2, 0], s_y)
            z_val = atan2(rmat[1, 0], rmat[0, 0])
        else:
            x_val = atan2(-rmat[1, 2], rmat[1, 1])
            y_val = atan2(-rmat[2, 0], s_y)
            z_val = 0

        return np.array([x_val, y_val, z_val])
