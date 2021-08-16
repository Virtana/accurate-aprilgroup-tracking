"""
Camera Calibration

Approximately > 15 images are taken of different angles
and rotations of a chessboard. These images are used to
calibrate the camera. If camera intrinsics already exists,
it loads them, else it uses the images taken to initiate calibration.

Important when taking images:
1. Ensure that the chessboard is on a flat surface. You want to
    get the distortions caused by the camera, not the object.
2. Take many pictures as necessary.
3. Take pictures with different angles to improve the quality of
    the calibration.
4. Use an environment with good illumination.

"""

import os
from pathlib import Path
from logging import Logger
from typing import List, Tuple
import numpy as np
import cv2


class Calibration:
    """Performs Camera Calibration using a Chessboard.

    Using photos of different rotations and translations of a chessboard,
    the camera is calibrated. If camera intrinsics already exists,
    it loads them, else it uses the images taken to initiate calibration.

    Attributes:
        logger: Used to create class specific logs for info and debugging.
        mtx: Camera Matrix.
        dist: Camera Distortion Coefficients.
        rvecs: Rotation Vector.
        tvecs: Translation Vector.
        chessboard_size: Size (Width and Height) of ChessBoard.
        frame_size: Pixel width and height of camera frame.
        criteria: Termination Criteria.
        objpoints: 3d point in real world space.
        imgpoints: 2D points in image plane.
    """

    _INTRINSIC_PARAMETERS_FILE = "aprilgroup_tracking/calibration/CameraParams.npz"
    _IMAGES_FOLDER = "aprilgroup_tracking/calibration/images_usb"

    def __init__(self, logger: Logger) -> None:
        self.logger = logger
        self.mtx: np.ndarray
        self.dist: np.ndarray
        self.rvecs: np.ndarray
        self.tvecs: np.ndarray
        self.chessboard_size: Tuple[int, int] = (9, 6)  # Width, Height
        self.frame_size: Tuple[int, int] = (1280, 720)  # Pixel size of Camera
        self.criteria: Tuple[float, int, float]
        self.objpoints: List[object] = []
        self.imgpoints: List[object] = []

    def try_load_intrinsic(self) -> bool:
        """Load intrinsics file if it is already there.

        Returns:
        Returns True if loaded, false if not.
        """

        try:
            self.logger.info("Trying to retrieve last \
                intrinsic calibration parameters.")
            self.load_intrinsic()
        except IOError:
            self.logger.info("Could not load previous intrinsic parameters.")
            return False
        self.logger.info("Loaded previous intrinsic parameters.")

        return True

    def start_intrinsic_calibration(self) -> np.ndarray:
        """Initiate calibration of the camera to obtain intrinsic values.

        Returns:
        Object Points of chessboard.
        """

        # Termination Criteria
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30, 0.001)

        # Prepare object points, e.g. (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp: np.ndarray = np.zeros(
            (self.chessboard_size[0] * self.chessboard_size[1], 3),
            np.float32)
        objp[:, :2] = np.mgrid[
            0:self.chessboard_size[0],
            0:self.chessboard_size[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.
        self.logger.info("Starting Intrinsic calibration.")

        return objp

    def save_intrinsic(self) -> None:
        """Save Camera Parameters."""

        np.savez(self._INTRINSIC_PARAMETERS_FILE,
                 mtx=self.mtx,
                 dist=self.dist,
                 rvecs=self.rvecs,
                 tvecs=self.tvecs)

    def load_intrinsic(self) -> None:
        """Loads the Camera Parameters"""
        with np.load(self._INTRINSIC_PARAMETERS_FILE) as file:
            self.mtx, self.dist, self.rvecs, self.tvecs = [file[i] for i in (
                'mtx',
                'dist',
                'rvecs',
                'tvecs')]

    def calculate_intrinsic(self) -> None:
        """
        Calculate the intrisic values of the camera based on the
        chessboard size, criteria and object points.
        Save them to a .npz file.
        """

        # Start the camera calibration
        objp = self.start_intrinsic_calibration()

        # Get all images in directory here.
        dirpath = self._IMAGES_FOLDER

        # Loop through all images, do oberations on images and
        # store the image points to do calibration
        # and undistort the images later on.
        for path, _, files in os.walk(dirpath):
            for file in files:
                filepath = Path(path) / file
                img = cv2.imread(str(filepath))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(
                    gray,
                    self.chessboard_size,
                    None)

                # If found, add object points,
                # image points (after refining them)
                if ret:

                    self.objpoints.append(objp)
                    # Finding corners in sub pixels
                    corners2 = cv2.cornerSubPix(
                        gray,
                        corners,
                        (11, 11),
                        (-1, -1),
                        self.criteria)
                    self.imgpoints.append(corners)

                    # Draw and display the corners
                    cv2.drawChessboardCorners(
                        img,
                        self.chessboard_size,
                        corners2,
                        ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(1000)  # Wait 1 sec, then go to other image.

            cv2.destroyAllWindows()

        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            self.frame_size,
            None,
            None)

        self.logger.info("Camera Calibrated: {}".format(ret))
        self.logger.info("\nCamera Matrix:\n {}".format(self.mtx))
        self.logger.info("\nDistortion Parameters:\n {}".format(self.dist))
        self.logger.info("\nRotation Vectors: \n {}".format(self.rvecs))
        self.logger.info("\nTranslation Vectors: \n {}".format(self.tvecs))

        # Save intrinsic parameters to .npz file
        self.save_intrinsic()
        self.get_reprojection_error()
        self.logger.info("Intrinsic Parameters have been saved!")

    def get_reprojection_error(self) -> None:
        """
        Obtains the reprojection error after camera calibration.
        The lower the error, the better calibrated the camera.
        Values should range from ...
        """
        # Reprojection Error
        mean_error = 0
        errors = []

        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i],
                self.rvecs[i],
                self.tvecs[i],
                self.mtx,
                self.dist)
            error = cv2.norm(
                self.imgpoints[i],
                imgpoints2,
                cv2.NORM_L2)/len(imgpoints2)
            errors.append(error)
            mean_error += error

        self.logger.info("Errors: {}".format(
            errors))
        self.logger.info("Total error: {}".format(
            mean_error/len(self.objpoints)))
