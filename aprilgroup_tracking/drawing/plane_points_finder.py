"""
Obtains the points on a plane for plane calibration.
"""

import os
import numpy as np
import cv2
import apriltag
from pathlib import Path
from typing import List, Tuple
import random
from aprilgroup_pose_estimation.transform_helper import TransformHelper
from aprilgroup_pose_estimation.draw import Draw


class PlanePointsFinder(TransformHelper, Draw):
    """Obtains world coordinates of points on a fixed plane for plane calibration.

    Attributes:
        logger: Used to create class specific logs for info and debugging.
        mtx: Camera Matrix.
        dist: Camera Distortion Coefficients.
        img: Original Frame.
        options: AprilTag Detectos Options.
    """

    _IMAGES_FOLDER = "aprilgroup_tracking/drawing/images_points"
    _WORLD_COORDS_FILE = "aprilgroup_tracking/drawing/WorldCoords.npz"

    def __init__(self, logger, mtx, dist):
        """Inits PlaneCalibrator Class with a logger, camera matrix and 
        distortion coefficients, the AprilTag size used, the world coordinate
        list and AprilTag Detector Options.
        """

        TransformHelper.__init__(self, logger, mtx, dist)
        Draw.__init__(self, logger)

        self.logger = logger
        self.mtx: np.ndarray = mtx              # Camera matrix
        self.dist: np.ndarray = dist            # Camera distortions
        self.tag_size: float = 0.017            # Size of AprilTags used to obtain the points
        self.world_coords_list = []
        self.img: np.ndarray                    # Original Frame

        # AprilTag detector options
        self.options = apriltag.DetectorOptions(families='tag36h11',
                                        border=1,
                                        nthreads=4,
                                        quad_decimate=1.0,
                                        quad_blur=0.0,
                                        refine_edges=True,
                                        refine_decode=False,
                                        refine_pose=True,
                                        debug=False,
                                        quad_contours=True)

    def try_load_world_coords(self) -> bool:
        """Load worlds coords file if it is already there.

        Returns:
        Returns True if loaded, false if not.
        """

        try:
            self.logger.info("Trying to retrieve last \
                world coordinates.")
            self.load_world_coords()
        except IOError:
            self.logger.info("Could not load previous world coordinates.")
            return False
        self.logger.info("Loaded previous world coordinates.")

        return True

    def get_points(self):
        """
        """

        # Get all images in directory here.
        dirpath = self._IMAGES_FOLDER

        # Loop through all images, do operations on images and
        # store the image points to do calibration
        # and undistort the images later on.
        for path, _, files in os.walk(dirpath):
            for file in files:
                filepath = Path(path) / file

                img = cv2.imread(str(filepath))
                self.img = self._process_frame(img)
                gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

                img_points_arr, obj_points_arr, tag_ids = self._obtain_initial_points(gray)
                transform = self._obtain_transform(img_points_arr, obj_points_arr)
                world_coords = self.transform_marker_corners(obj_points_arr, transform).reshape(-1, 3)

                # Save a random point as a point on the plane
                self.world_coords_list.append(world_coords[random.randint(0, 2)])

                cv2.imshow('img', self.img)
                cv2.waitKey(1000)  # Wait 1 sec, then go to other image.

            cv2.destroyAllWindows()

        self.save_world_coords()

    def save_world_coords(self) -> None:
        """Save Camera Parameters."""

        np.savez(self._WORLD_COORDS_FILE,
                 world_coords=self.world_coords_list)

    def load_world_coords(self) -> None:
        """Loads the World Coords"""
        with np.load(self._WORLD_COORDS_FILE, allow_pickle=True) as file:
            self.world_coords_list = file['world_coords']

    def _obtain_initial_points(self, gray:np.ndarray):
        """Obtains the tag ids detected, along with their image and object points.
        
        Args:
        self.mtx: 
            Camera Matrix.
        gray: 
            Frame turned to grayscale (only preprocessing needed to properly detect AprilTags)

        Returns:
        A list of all Image points, and their corresponding list of object points and tag ids.
        """

        detector = apriltag.Detector(self.options)

        # Detect the apriltags in the image
        detection_results = detector.detect(gray)

        # Amount of april tags detected
        num_detections = len(detection_results)
        # self.logger.info('Detected {} tags.\n'.format(num_detections))

        img_points_arr = []
        obj_points_arr = []
        tag_ids = []

        # If 1 or more apriltags are detected, estimate and draw the pose
        if num_detections > 0:
            # If the camera was calibrated and the matrix is supplied
            if self.mtx is not None:
                for i, detection in enumerate(detection_results):
                    
                    # The higher decision margin, the better the detection (i.e. means more contrast within the tag).
                    if detection.decision_margin < 50:
                        continue

                    # Image points are the corners of the apriltag
                    imagePoints = detection.corners.reshape(1,4,2) 

                    # Draw square on all AprilTag edges
                    self.draw_corners(self.img, detection)

                    objpts = self.get_initial_pts(self.tag_size)

                    img_points_arr.append(imagePoints)
                    tag_ids.append(detection.tag_id)
                    obj_points_arr.append(objpts)

        return img_points_arr, obj_points_arr, tag_ids

    def _obtain_transform(self, img_points_arr: List[np.ndarray], obj_points_arr: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Obtains the pose of the AprilGroup.

        Args:
        img_points_arr:
            Image points List obtained from all detected AprilTags in a frame.
        obj_points_arr:
            Respective Object Points List for each image points obtained.

        Returns:
        The pose of obtained with the points overlay
        onto the AprilGroup and the 3D drawing to observe
        and track the AprilGroup.
        """

        transformation = (np.array([]), np.array([]))

        if img_points_arr and obj_points_arr:
            obj_points_arr = np.array(obj_points_arr, dtype=np.float32).reshape(-1, 3)  # Nx3 array
            img_points_arr = np.array(img_points_arr, dtype=np.float32).reshape(-1, 2)  # Nx2 array

            # Obtain the pose of the apriltag
            # If the last pose is None, obtain the pose with
            # no Extrinsic Guess, else use Extrinsic guess and the last pose
            success, pose_rvecs, pose_tvecs = cv2.solvePnP(
                    obj_points_arr,
                    img_points_arr,
                    self.mtx,
                    self.dist,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

            # If pose was found successfully
            if success:
                mean_error = self.get_reprojection_error(obj_points_arr, img_points_arr, pose_rvecs, pose_tvecs)
                # self.logger.info("mean error: {} \n pose rvec: {} \n pose tvec: {}".format(mean_error, pose_rvecs, pose_tvecs))

                # Increased for optical flow but optimal one is <1
                if mean_error < 2:

                    transformation = (pose_rvecs, pose_tvecs)

                    self.logger.info("Projecting 3D points onto the image plane.")
                    # Project the 3D points onto the image plane
                    imgpts, jac = cv2.projectPoints(obj_points_arr, transformation[0], transformation[1], self.mtx, self.dist)
                    
                    self.logger.info("Drawing the points...")
                    # Draw the image points overlay onto the object and the 3D Drawing
                    self.draw_squares_and_3d_pts(self.img, None, imgpts, (255, 255, 255))

        return transformation

    def _process_frame(self, frame:np.ndarray) -> np.ndarray:
        """Undistorts Frame (other processing, if needed, can go here)

        Args:
        frame: 
            Each frame from the camera.

        Returns:
        Processed Frame.
        """

        # Undistorts the frame
        if self.dist is not None:
            frame = self.undistort_frame(frame, self.mtx, self.dist)
        
        return frame
