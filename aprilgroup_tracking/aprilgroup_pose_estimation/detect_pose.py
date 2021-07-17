"""
Detects and Estimates the Pose of the AprilGroup attached to a Dodecahedron.
"""

import json
import numpy as np
import cv2
import apriltag
from pathlib import Path
import datetime
import time
from typing import List, Dict, Tuple
from numpy.testing import assert_array_almost_equal
from copy import deepcopy
# import the math module 
from math import sqrt, sin, cos, atan, degrees, radians, pi, atan2, asin


class DetectAndGetPose:
    """Detects AprilGroup and Obtains Pose of object with AprilGroup Attached.

    This Class detects AprilTags on an object (dodecahedron in this case).
    Using these detections, an AprilGroup is formed when more than 1 tag is
    detected. The image points (corners of the AprilTags), are obtained, along
    with the corresponding object points from predefined extrinsics obtained
    during the calibration of the dodecahedron.

    Using the image and object points, the pose of the dodecahedron is
    obtained, and the 3D points are projected onto the object in the frame.
    A 3D drawing of the dodecahedron is also displayed.

    Attributes:
        logger: Used to create class specific logs for info and debugging.
        mtx: Camera Matrix.
        dist: Camera Distortion Coefficients.
        img: Original Frame.
        draw_frame: 3D Black Drawing Frame.
        prev_transform: Previous Pose of Dodecahedron initialised to None.
        options: AprilTag Detectos Options.
        extrinsics: Tag sizes, translation and rotation vectors for
                    predefined AprilGroup.
        opointsArr: AprilGroup Object Points.
    """

    DIRPATH = 'aprilgroup_tracking/aprilgroup_pose_estimation'
    JSON_FILE = 'april_group.json'

    def __init__(self, logger, mtx, dist):
        """Inits DetectAndGetPose Class with a logger,
        camera matrix and distortion coefficients, the
        two frames to be displayed, AprilTag Detector Options,
        the predefined AprilGroup Extrinsics and the AprilGroup
        Object Points.
        """
        self.logger = logger
        self.mtx: np.ndarray = mtx              # Camera matrix
        self.dist: np.ndarray = dist            # Camera distortions
        self.img: np.ndarray                    # Original Frame
        self.draw_frame: np.ndarray             # 3D Drawing Frame
        # Previous Pose of Dodecahedron.
        self.prev_transform: Tuple(np.ndarray, np.ndarray) = (None, None)
        self.extrinsic_guess: Tuple(np.ndarray, np.ndarray) = (None, None)

        self.rot_velocities = []
        self.tran_velocities = []

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

        # Get all extrinsics (tag size, translation and rotation vectors
        # for predefined aprilgroup) from the .json file
        self.extrinsics = self.get_extrinsics()

        # Extract the tag size, rvec and tvec for all apriltags on dodeca
        # and obtain the aprilgroup object points
        self.opointsArr = self.get_all_points(self.extrinsics)

    def add_values_in_dict(
        self,
        sample_dict: Dict,
        key: int,
        list_of_values: List[object]
    ) -> Dict:
        """Appends multiple values to a key in the given dictionary

        Args:
        sample_dict:
            Dictionary to add the values to.
        key:
            The key in the dictionary.
        list_of_values:
            The values to be assigned to a specific key.

        Returns:
        A dict mapping keys to the corresponding data
        fetched. For example:

        {'key1': ('Key1 Data1', 'Key1 Data2'),
        'key2': ('Key2 Data1', 'Key2 Data2'),
        'key3': ('Key3 Data1', 'Key4 Data2')}

        """

        if key not in sample_dict:
            sample_dict[key] = list()
        sample_dict[key].extend(list_of_values)

        return sample_dict

    def get_extrinsics(self) -> Dict:
        """Obtains the tag sizes, rvecs and tvecs for each apriltag
        from the .json file

        Args:
        JSON_FILE:
            The json file to be used to obtain the extrinsics.

        Returns:
        A dict with all the extrinsic values paired to their
        corresponding tag_id. For example:

        {
            '205': (0.017834, [[0.443 0.122 0.124]], [[0.755 0.1684 0.074]]),
            '200': (0.018393, [[0.094 0.353 0.775]], [[0.0024 0.0544 0.09]])
        }

        """

        # Extrinsics Dict
        extrinsics: Dict = {}

        # Opening json file containing the dodecahedron extrinsics
        filepath = Path(self.DIRPATH) / self.JSON_FILE
        with open(filepath, "r") as f:
            data = json.load(f)

        for key, tags in data['tags'].items():
            # Size of the tags
            tag_sizes = tags['size']
            # Turning tvec into N x 1 array
            tvecs = np.array(tags['extrinsics'][:3],
                             dtype=np.float32).reshape((3, 1))
            # Turning rvec into N x 1 array
            rvecs = np.array(tags['extrinsics'][-3:],
                             dtype=np.float32).reshape((3, 1))
            # Add extrinsics to their specific tag_id
            extrinsics = self.add_values_in_dict(
                                                 extrinsics,
                                                 int(key),
                                                 [tag_sizes,
                                                  tvecs,
                                                  rvecs]
                                                )

        # Closing file
        f.close()

        self.logger.info('Successfully Loaded AprilGroup Extrinsics!')

        return extrinsics

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Undistorts the camera frame given the camera matrix and
        distortion values from camera calibration

        Args:
        frame:
            Current camera frame.
        self.mtx:
            Camera Matrix.
        self.dist:
            Distortion Coefficients from Calibrated Camera.

        Returns:
        A Numpy Array that contains the undistorted frame.
        """

        # Height and Width of the camera frame
        h,  w = frame.shape[:2]

        # Get the camera matrix and distortion values
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
                                                             self.mtx,
                                                             self.dist,
                                                             (w, h),
                                                             1,
                                                             (w, h)
                                                            )

        # Undistort Frame
        dst = cv2.undistort(frame, self.mtx, self.dist, None, newCameraMatrix)

        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        return dst

    def draw_boxes(self, imgpts: np.ndarray) -> None:
        """Draws the lines and edges on the april tag images
        to show the pose estimations.

        Args:
        self.img:
            The image containing dodecahedron with apriltags attached.
        self.imgpts:
            Image points of the AprilGroup returned from cv2:ProjectPoints().
        self.dist:
            Distortion Coefficients from Calibrated Camera.

        Returns:
        3D Boxes that shows the AprilGroup detected and the pose estimated.
        """

        # Bounding box for AprilTag, this will display a
        # 3D cube on detected AprilTags
        # in the pose direction
        edges = np.array([
            0, 1,
            1, 2,
            2, 3,
            3, 0,
            0, 4,
            1, 5,
            2, 6,
            3, 7,
            4, 5,
            5, 6,
            6, 7,
            7, 4
        ]).reshape(-1, 2)

        # Overlay Pose onto image
        imgpts = np.round(imgpts).astype(int)
        imgpts = [tuple(pt) for pt in imgpts.reshape(-1, 2)]

        # Draws lines within the edges given
        for i, j in edges:
            cv2.line(self.img, imgpts[i], imgpts[j], (0, 255, 0), 1, 16)

    def draw_squares_and_3d_pts(self, imgpts: np.ndarray) -> None:
        """Extracts the bounding box (x, y)-image points
        returned from cv2:projectPoints() for the AprilGroup
        and convert each of the (x, y)-coordinate pairs to integers.

        Args:
        self.img:
            Original frame data.
        self.draw_frame:
            Second window to display 3D drawing of the Dodecahedron.
        imgpts:
            Image points returned from cv2:projectPoints
            (mapping 3D to 2D points).

        Returns:
        Bounding box to form a 3D Dodecahedron drawing,
        and image points overlay on the AprilGroup Detected.
        """

        # Overlay Pose onto image
        ipoints = np.round(imgpts).astype(int)
        ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]

        # Draw points obtained from cv2:projectPoints()
        # overlay onto the dodecahedron object itself.
        for i in ipoints:
            if i[1] < 720 and i[0] < 1280:
                cv2.circle(self.img, (i[0], i[1]), 5, (0, 0, 255), -1)

        # Obtain the 4 points from the image points
        length = len(imgpts)
        for i in range(0, length, 4):
            (ptA, ptB, ptC, ptD) = imgpts[i:i+4].reshape(-1, 2)
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))

            # Draw the 3D form of the dodecahedron from the image points
            # obtained on a second frame
            cv2.line(
                self.draw_frame,
                ptA, ptB, (255, 255, 255),
                5, cv2.LINE_AA)
            cv2.line(
                self.draw_frame,
                ptB, ptC, (255, 255, 255),
                5, cv2.LINE_AA)
            cv2.line(
                self.draw_frame,
                ptC, ptD, (255, 255, 255),
                5, cv2.LINE_AA)
            cv2.line(
                self.draw_frame,
                ptD, ptA, (255, 255, 255),
                5, cv2.LINE_AA)

    def draw_corners(self, detection: apriltag.Detection) -> None:
        """Extracts the bounding box (x, y)-coordinates for the AprilTag
        and convert each of the (x, y)-coordinate pairs to integers.

        Args:
        self.img:
            Original frame data.
        detections:
            AprilTag detections found via the AprilTag library.

        Returns:
        Bounding box with center point and tag id shown overlay
        on each AprilTag detection.
        """

        # For all detections, get the corners and draw the
        # bounding box, center and tag id

        # AprilTag Corners (Image Points)
        (ptA, ptB, ptC, ptD) = detection.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))

        # Draw the bounding box of the AprilTag detection
        cv2.line(self.img, ptA, ptB, (0, 255, 0), 5)
        cv2.line(self.img, ptB, ptC, (0, 255, 0), 5)
        cv2.line(self.img, ptC, ptD, (0, 255, 0), 5)
        cv2.line(self.img, ptD, ptA, (0, 255, 0), 5)

        # Draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(detection.center[0]), int(detection.center[1]))
        cv2.circle(self.img, (cX, cY), 5, (0, 255, 255), -1)

        # Draw the tag family on the image
        tag_id = "ID: {}".format(detection.tag_id)
        cv2.putText(self.img, tag_id, (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def draw_contours(self, imgpts: np.ndarray) -> None:
        """Draws the contour shape of the image onto the second openCV window.

        Args:
        draw_frame:
            Second window to display 3D drawing of the Dodecahedron.
        imgpts:
            Coordinates of 3D points projected on 2D image plane.

        Returns:
        3-Dimensional shape of the image drawn on the second window.
        """

        # Overlay Pose onto image
        ipoints = np.round(imgpts).astype(int)
        ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]

        # Draw 3-dimensional shape of the image
        a = np.array(ipoints)
        cv2.drawContours(self.draw_frame, [a], 0, (255, 255, 255), -1)

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
        mrv, jacobian = cv2.Rodrigues(transformation[0])

        # Apply rotation to 3D points in space
        app_rot_mat = object_pts @ mrv.T

        # Apply Translation to 3D points in space
        tran_vec = transformation[1].reshape(-1, 3)
        tran_mat = app_rot_mat + tran_vec

        return tran_mat

    def get_all_points(self, extrinsics: Dict) -> np.ndarray:
        """Get all the points from the .json file and obtain the object points.

        Args:
        extrinsics:
            Dict that contains the key to extrinsics
            data pairings for all tags on the dodecahedron.

        Returns:
        The 3D points in space of the entire AprilGroup.
        """

        obj_points = []

        for i in extrinsics:
            # Dict key is tag id, with the tag size,
            # tvec, rvec, respectively, appended
            tag_size = extrinsics[i][0]

            # Tuple with rvec (first in tuple) and
            # tvec (second in tuple) transformation
            transformation = (extrinsics[i][2], extrinsics[i][1])

            # 3D Initial Marker Points
            initial_obj_pts = self.get_initial_pts(tag_size)

            # Obtain 3D points in space for AprilGroup
            marker_corners = self.transform_marker_corners(
                                                           initial_obj_pts,
                                                           transformation
                                                          )
            obj_points.append(marker_corners)

        # Form needed to pass the object points into
        # cv2:solvePnP() and cv2:projectPoints()
        opointsArr = np.array(obj_points).reshape(-1, 3)

        return opointsArr

    def _update_buffers(self, rot_vel, tran_vel, buf_size=2):
        """
        Adds rotational and translational velocities
        to their respective buffers. Will remove old
        items in the buffers based on buf_size
        """
        self.rot_velocities.append(rot_vel)
        self.tran_velocities.append(tran_vel)
        if len(self.rot_velocities) > buf_size:
            self.rot_velocities.pop(0)
            self.tran_velocities.pop(0)

    def get_relative_trans(self, prev_rot_mat, t1, t0):
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
        tran_vel = (prev_rot_mat.T).dot(t1) - (prev_rot_mat.T).dot(t0)
        # tran_vel = (prev_rot_mat.T).dot(t0) - (prev_rot_mat.T).dot(t1)

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

        r0_to_r1 = r1.dot(r0.T)
        # r0_to_r1 = r0.dot(r1.T)

        # Verify correctness: apply r0_to_r1 after r_0
        assert_array_almost_equal(r1, r0_to_r1.dot(r0))
        # assert_array_almost_equal(r0, r0_to_r1.dot(r1))

        return r0_to_r1

    def get_pose_vel_acc(self, curr_rvecs, prev_rvecs, curr_tvecs, prev_tvecs):
        
        # TODO: Optimise this funciton
        success = False

        # Obtain the rotation matrices 
        prev_rmat, jac = cv2.Rodrigues(prev_rvecs)
        curr_rmat, jac = cv2.Rodrigues(curr_rvecs)

        # Translational Velocity
        tran_vel = self.get_relative_trans(prev_rmat, curr_tvecs, prev_tvecs)

        # Rotational Velocity
        rot_vel = self.get_relative_rot(prev_rmat, curr_rmat)

        # Add velocities to their respective buffers
        self._update_buffers(rot_vel, tran_vel)

        # ACCELERATION (Constant Acceleration)
        # TODO: find length of translation or rotation, do some error checking
        vel_len = len(self.tran_velocities)
        if vel_len == 1:
            tran_acc = self.tran_velocities[0]
            rot_acc = self.rot_velocities[0]
        elif vel_len > 1:
            tran_acc = self.get_relative_trans(self.rot_velocities[vel_len-2], self.tran_velocities[vel_len-1], self.tran_velocities[vel_len-2])
            rot_acc = self.get_relative_rot(self.rot_velocities[vel_len-2], self.rot_velocities[vel_len-1])

        if vel_len:
            success = True

        return success, tran_vel, rot_vel, tran_acc, rot_acc

    def apply_vel_acc(self, rvec, tvec, tran_vel, tran_acc, rot_vel, rot_acc):
        """Applies the pose velocity and acceleration to the last pose.

        Args:
        rvec:
        tvec:
        tran_vel:
        tran_acc:
        rot_vel:
        rot_acc:

        Returns:
        The predicted pose (rotation and translation)
        """
        # Equation used: (Last pose + pose velocity + 0.05*pose acceleration)

        # Predicted translation
        self.logger.info("\n Translation: {} \n Translational Velocity: {} \n Acceleration: {}:".format(tvec, tran_vel, tran_acc))

        tvec_pose = (tvec + tran_vel) + (0.5*tran_acc)

        # Prediction rotation
        rot_mat, jac = cv2.Rodrigues(rvec)
        rmax_pose = (rot_mat + rot_vel) + (0.5*rot_acc)
        rvec_pose, jac = cv2.Rodrigues(rmax_pose)

        return (rvec_pose, tvec_pose)

    def _obtain_detections(
        self,
        gray: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Obtains the tag ids detected, along with
        their image and object points.

        Args:
        self.mtx:
            Camera Matrix.
        gray:
            Frame turned to grayscale (only preprocessing
            needed to properly detect AprilTags)

        Returns:
        A list of all Image points, and their corresponding
        list of object points and tag ids.
        """

        detector = apriltag.Detector(self.options)

        # Detect the apriltags in the image
        detection_results, dimg = detector.detect(gray, return_image=True)

        # Amount of april tags detected
        num_detections = len(detection_results)
        self.logger.info('Detected {} tags.\n'.format(num_detections))

        imgPointsArr = []
        objPointsArr = []
        tag_ids = []

        # If 1 or more apriltags are detected, estimate and draw the pose
        if num_detections > 0:
            # If the camera was calibrated and the matrix is supplied
            if self.mtx is not None:
                for i, detection in enumerate(detection_results):

                    # The higher decision margin, the better the detection
                    # (i.e. means more contrast within the tag).
                    if detection.decision_margin < 50:
                        continue

                    self.logger.info(
                        "Detection {} of {}:".format(
                                                     i + 1,
                                                     num_detections
                                                    ))
                    self.logger.info("\n" + detection.tostring(indent=2))

                    # Image points are the corners of the apriltag
                    imagePoints = detection.corners.reshape(1, 4, 2)

                    # Draw square on all AprilTag edges
                    self.draw_corners(detection)

                    # Obtain the extrinsics from the .json file
                    # for the first apriltag detected
                    # Size of the AprilTag markers
                    markersize = self.extrinsics[detection.tag_id][0]

                    # Tuple with rvec (Rotation Vector of AprilTag) and
                    # tvec (Translation Vector of AprilTag) transformation
                    transformation = (
                        self.extrinsics[detection.tag_id][2],
                        self.extrinsics[detection.tag_id][1]
                    )

                    # Obtains the initial 3D points in space
                    # for each detected AprilTag
                    initial_obj_pts = self.get_initial_pts(markersize)
                    # Obtain the object points (marker_corners)
                    # of the apriltag detected via rotating and
                    # translating the AprilGroup
                    objpts = self.transform_marker_corners(
                        initial_obj_pts,
                        transformation
                    )

                    imgPointsArr.append(imagePoints)
                    objPointsArr.append(objpts)
                    tag_ids.append(detection.tag_id)

        return imgPointsArr, objPointsArr, tag_ids

    def _project_draw_points(
        self,
        transformation: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Projects the 3D points onto the image plane and draws those points.

        Args:
        prvecs:
            Rotation pose vector from cv2:solvePnP().
        ptvecs:
            Translation pose vector from cv2:solvePnP().

        Returns:
            Obtains the 3D points projected onto the image plane to be drawn.
        """

        # Project the 3D points onto the image plane
        imgpts, jac = cv2.projectPoints(
            self.opointsArr,
            transformation[0],
            transformation[1],
            self.mtx,
            self.dist
        )

        self.logger.info("Drawing the points...")
        # Draw the image points overlay onto the object and the 3D Drawing
        self.draw_squares_and_3d_pts(imgpts)

    def _estimate_pose(
        self,
        imgPointsArr: List[np.ndarray],
        objPointsArr: List[np.ndarray]
    ) -> None:
        """Obtains the pose of the dodecahedron.

        Args:
        self.mtx:
            Camera Matrix.
        self.dist:
            Camera Distortion Parameters.
        imgPointsArr:
            Image points List obtained from all detected AprilTags in a frame.
        objPointsArr:
            Respective Object Points List for each image points obtained.

        Returns:
        The pose of obtained with the points overlay
        onto the dodecahedron and the 3D drawing to observe
        and track the dodecahedron.
        """

        if imgPointsArr and objPointsArr:
            objPointsArr = np.array(objPointsArr).reshape(-1, 3)  # Nx3 array
            imgPointsArr = np.array(imgPointsArr).reshape(-1, 2)  # Nx2 array

            # Obtain the pose of the apriltag
            # If the last pose is None, obtain the pose with
            # no Extrinsic Guess, else use Extrinsic guess and the last pose
            if self.extrinsic_guess[0] is None:
                success, pose_rvecs, pose_tvecs = cv2.solvePnP(
                    objPointsArr,
                    imgPointsArr,
                    self.mtx,
                    self.dist,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            else:
                success, pose_rvecs, pose_tvecs = cv2.solvePnP(
                    objPointsArr,
                    imgPointsArr,
                    self.mtx,
                    self.dist,
                    self.extrinsic_guess[0],
                    self.extrinsic_guess[1],
                    True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

            transformation = (pose_rvecs, pose_tvecs)
            self.logger.info("Pose Obtained {}:".format(transformation))

            # If pose was found successfully
            if success:
                # Can not be 'behind' barcode, or too far away
                if transformation[1][2][0] > 0 and transformation[1][2][0] < 1000:
                    self.logger.info("Projecting 3D points onto the image plane.")
                    # Project the 3D points onto the image plane
                    self._project_draw_points(transformation)

                    # If this is the second frame, there would be no velocity or acc calculation
                    if self.prev_transform[0] is None:
                        # Assign the previous pose to current pose to obtain last pose
                        # Used as an extrinisic guess
                        self.extrinsic_guess = transformation
                        # Obtain the last pose (used in the calculation for predicted pose)
                        self.prev_transform = transformation
                    else:
                        good, tran_vel, rot_vel, tran_acc, rot_acc = self.get_pose_vel_acc(transformation[0], self.prev_transform[0], transformation[1], self.prev_transform[1])
                        if good:
                            self.logger.info("Obtained pose velocity and acceleration!")
                            pred_transform = self.apply_vel_acc(self.prev_transform[0], self.prev_transform[1], tran_vel, tran_acc, rot_vel, rot_acc)
                            self.logger.info("Predicted Transform {}:".format(pred_transform))

                            # Assign the previous pose to predicted pose
                            self.extrinsic_guess = pred_transform
                            # Obtain the last pose (used in the calculation for predicted pose)
                            self.prev_transform = transformation
            else:
                # Clear iteration if SolvePNP is 'bad'
                pose_rvecs = None
                pose_tvecs = None

    def _detect_and_get_pose(self, frame: np.ndarray) -> None:
        """Obtains the pose of the dodecahedron.

        Obtains each frame from the camera,
        uses the apriltag library to detect the apriltags,
        overlays on the apriltag, and uses those detections to
        estimate the pose using OpenCV functions
        cv2:solvePnP() and cv2:projectPoints().

        Args:
        frame:
            Each frame from the camera

        Returns:
        Pose and points of the AprilGroup overlayed on the dodecahedron
        object and the 3D drawing of the dodecahedron in a seperate window.
        """

        # Get the frame from the Video
        self.img = frame

        # Form a black frame to display the 3D drawing of the dodecahedron
        h,  w = self.img.shape[:2]
        self.draw_frame = np.zeros(shape=[h, w, 3], dtype=np.uint8)

        # Apply grayscale to the frame to get proper AprilTag Detections
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Obtain AprilGroup Detected and their respective
        # tag ids, image and object points
        imgPointsArr, objPointsArr, tag_ids = self._obtain_detections(gray)

        if tag_ids is not None:
            # Using those points, estimate the pose of the dodecahedron
            self._estimate_pose(imgPointsArr, objPointsArr)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Undistorts Frame (other processing, if needed, can go here)

        Args:
        frame:
            Each frame from the camera.

        Returns:
        Processed Frame.
        """

        # Undistorts the frame
        if self.dist is not None:
            frame = self.undistort_frame(frame)

        return frame

    def overlay_camera(self) -> None:
        """

        Creates a new camera window, to show both the pose estimation boxes
        drawn on the apriltags, and projections overlayed onto the
        incoming frames. This will allow us to quickly see the limitations
        of the baseline approach, i.e. when detection fails.

        Args:
        frame:
            Each frame from the camera

        Returns:
        Two video captured windows, one displays the object and the points
        returned from the pose, overlaid over the object, the
        other displays the 3D drawing of the object pose.
        """

        # Create a cv2 window to show images
        window = 'Camera'
        cv2.namedWindow(window)

        # Open the first camera to get the video stream and the first frame
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        time.sleep(2.0)
        success, frame = cap.read()

        if success:
            frame = self.process_frame(frame)
            # Obtains the pose of the object on the
            # frame and overlays the object.
            self._detect_and_get_pose(frame)

        while True:

            success, frame = cap.read()

            if success:
                frame = self.process_frame(frame)
                # Obtains the pose of the object on the
                # frame and overlays the object.
                self._detect_and_get_pose(frame)
            else:
                break

            # draw the text and timestamp on the frame
            cv2.putText(
                frame,
                "Displaying dodecahedron with aprilgroup".format(),
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1
            )

            # Display the object itself with points overlaid onto the object
            cv2.imshow(window, self.img)
            # Display the black frame window that shows a
            # 3D drawing of the object
            cv2.imshow('image', self.draw_frame)

            # if ESC clicked, break the loop
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                break
