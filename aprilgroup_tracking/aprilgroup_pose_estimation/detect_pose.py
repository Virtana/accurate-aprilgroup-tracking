"""This module detects AprilTags and estimates their poses.

TODO: Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

    Typical usage example:

    To use flow, set useflow to True.
    If the user wishes to test APE alone, set useflow to False.

        useflow = True
        det_pose = DetectAndGetPose(det_pose_logger, mtx, dist)
        det_pose.overlay_camera(useflow)
"""

import json
import numpy as np
import cv2 as cv
import apriltag
from pathlib import Path
import datetime
import time
from typing import List, Dict, Tuple
from copy import deepcopy
# Import necessary classes
from aprilgroup_pose_estimation.transform_helper import TransformHelper
from aprilgroup_pose_estimation.draw import Draw
from aprilgroup_pose_estimation.optical_flow import OpticalFlow


class DetectAndGetPose(TransformHelper, Draw, OpticalFlow):
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
        all_objpts: AprilGroup Object Points.
    """

    DIRPATH = 'aprilgroup_tracking/aprilgroup_pose_estimation'
    JSON_FILE = 'april_group.json'

    def __init__(self, logger, mtx, dist, enhance_ape):
        """Inits DetectAndGetPose Class with a logger,
        camera matrix and distortion coefficients, the
        two frames to be displayed, AprilTag Detector Options,
        the predefined AprilGroup Extrinsics and the AprilGroup
        Object Points.
        """

        TransformHelper.__init__(self, logger, mtx, dist)
        Draw.__init__(self, logger)
        OpticalFlow.__init__(self, logger)

        self.logger = logger
        self.mtx: np.ndarray = mtx              # Camera matrix
        self.dist: np.ndarray = dist            # Camera distortions
        self.img: np.ndarray                    # Original Frame
        self.draw_frame: np.ndarray             # 3D Drawing Frame
        # Previous Pose of Dodecahedron.
        self.prev_transform: Tuple(np.ndarray, np.ndarray) = (None, None)
        self.extrinsic_guess: Tuple(np.ndarray, np.ndarray) = (None, None)
        # Used to save rotational and translation velocities
        self.rot_velocities: List[object] = []
        self.tran_velocities: List[object] = []

        # Used to test pivot calibration
        self._rmats = []
        self._tvecs = []

        # Used to determine if APE should be used with 
        # no extrinsic guess or with the predicted pose 
        # as the extrinsic guess to enhance APE
        self.enhance_ape: bool = enhance_ape
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
        self.all_objpts = self.get_all_points(self.extrinsics)

    def get_extrinsics(self) -> Dict:
        """Obtains the tag sizes, rvecs and tvecs for each apriltag
        from the .json file and provides a dict with all the
        extrinsic values paired to their corresponding tag_id.
        """

        # Extrinsics Dict
        extrinsics: Dict = {}

        # Opening json file containing the dodecahedron extrinsics
        filepath = Path(self.DIRPATH) / self.JSON_FILE
        with open(filepath, "r") as f:
            data = json.load(f)

        for key, tags in data['tags'].items():
            try:
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
            except(RuntimeError, TypeError) as error:
                raise error
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

        try:
            # Height and Width of the camera frame
            h,  w = frame.shape[:2]

            # Get the camera matrix and distortion values
            newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(
                                                                self.mtx,
                                                                self.dist,
                                                                (w, h),
                                                                1,
                                                                (w, h)
                                                                )

            # Undistort Frame
            dst = cv.undistort(
                frame, self.mtx, self.dist, None, newCameraMatrix)

            # Crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
        except(RuntimeError, TypeError) as error:
                raise error

        return dst

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
            try:
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
            except(RuntimeError, TypeError) as error:
                raise error

        # Form needed to pass the object points into
        # cv:solvePnP() and cv:projectPoints()
        all_objpts = np.array(obj_points).reshape(-1, 3)

        return all_objpts

    def _update_buffers(self, rot_vel, tran_vel, buf_size=2) -> None:
        """
        Adds rotational and translational velocities
        to their respective queues. Will remove old
        items in the buffers based on buf_size
        """

        self.rot_velocities.append(rot_vel)
        self.tran_velocities.append(tran_vel)
        if len(self.rot_velocities) > buf_size:
            self.rot_velocities.pop(0)
            self.tran_velocities.pop(0)

    def get_pose_vel_acc(
        self,
        curr_transform,
        prev_transform
    ) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Obtains the pose velocity and acceleration.

        Args:
        curr_transform:
            Current frame's transformation.
        prev_transform:
            Previous frame's transformation.

        Returns:
        Success, the translational velocity and acceleration,
        and rotational velocity and acceleration.
        """

        self.logger.info(
            "Prev pose: rvecs: {}, \n tvecs: {}".format(
                prev_transform[0], prev_transform[1]))
        self.logger.info(
            "Curr pose: rvecs: {}, \n tvecs: {}".format(
                curr_transform[0], curr_transform[1]))

        success = False
        tran_vel = 0.0
        rot_vel = 0.0
        tran_acc = 0.0
        rot_acc = 0.0

        try:
            # Obtain the rotation matrices
            prev_rmat = cv.Rodrigues(prev_transform[0])[0]
            curr_rmat = cv.Rodrigues(curr_transform[0])[0]

            # Translational Velocity
            tran_vel = self.get_relative_trans(
                curr_rmat, curr_transform[1], prev_transform[1])

            # Rotational Velocity
            rot_vel = self.get_relative_rot(prev_rmat, curr_rmat)

            # Add velocities to their respective queues
            self._update_buffers(rot_vel, tran_vel)

            # Calculate Acceleration (based on Constant Acceleration)
            velocity_buf_len = len(self.tran_velocities)
            # If there are velocities in the buffer, you can find acceleration
            if velocity_buf_len > 1:
                success = True
                tran_acc = self.get_relative_trans(
                    self.rot_velocities[velocity_buf_len-1],
                    self.tran_velocities[velocity_buf_len-1],
                    self.tran_velocities[velocity_buf_len-2])
                rot_acc = self.get_relative_rot(
                    self.rot_velocities[velocity_buf_len-2],
                    self.rot_velocities[velocity_buf_len-1])
        except(RuntimeError):
                raise RuntimeError

        return success, tran_vel, rot_vel, tran_acc, rot_acc

    def apply_vel_acc(
        self,
        transformation,
        tran_vel,
        tran_acc,
        rot_vel,
        rot_acc
    ) -> np.ndarray:
        """Applies the pose velocity and acceleration to the last pose.

        Equation used: (Last pose + pose velocity + 0.05*pose acceleration)

        Args:
        transformation:
            Last frame's object transformation (rvec, tvec)
        tran_vel:
            Relative translation change between frames.
        tran_acc:
            Change between translational velocities.
        rot_vel:
            Relative rotation between frames.
        rot_acc:
            Change between rotational velocities.

        Returns:
        The predicted pose (rotation and translation)
        """

        try:
            # Obtain half rotational acceleration
            rot_acc_angle = self.rotation_matrix_to_euler_angles(rot_acc) / 2
            rot_acc = self.euler_angles_to_rotation_matrix(rot_acc_angle)

            # rvec needs to be rot matrix
            rmat = cv.Rodrigues(transformation[0])[0]
            self.logger.info(f"{rot_acc} {rot_vel} {rmat}")

            # Obtain the extrinsic matrix containing rvec and tvec
            extrinsic_pose = self.get_extrinsic_matrix(rmat, transformation[1])
            # Obtain the extrinsic matrix containing the pose velocities
            extrinsic_vel = self.get_extrinsic_matrix(rot_vel, tran_vel)
            # Obtain the extrinsic matrix containing the pose accelerations
            extrinsic_acc = self.get_extrinsic_matrix(rot_acc, 0.5*tran_acc)

            # Apply the pose velocities and accelerations to the last pose
            pred_pose = extrinsic_acc @ extrinsic_vel @ extrinsic_pose
            # Obtain the rmat and tvec from the extrinsic predicted pose
            rmat_pose, tvec_pose = self.get_rmat_tvec(pred_pose)
            rvec_pose = cv.Rodrigues(rmat_pose)[0]
        except(RuntimeError):
            raise RuntimeError

        self.logger.info(
            "\n POSE_GUESS: \n{}\n{}".format(rvec_pose, tvec_pose))

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
        tag_idsArr = []

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
                    try:
                        markersize = self.extrinsics[detection.tag_id][0]
                    except:
                        raise ValueError(
                            "An error occured when retrieving the markersize.")
                        continue

                    # Tuple with rvec (Rotation Vector of AprilTag) and
                    # tvec (Translation Vector of AprilTag) transformation
                    transformation = (
                        self.extrinsics[detection.tag_id][2],
                        self.extrinsics[detection.tag_id][1]
                    )

                    try:
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
                    except:
                        raise ValueError(
                            "An error occured trying to obtain \
                            the object points.")
                        continue

                    imgPointsArr.append(imagePoints)
                    objPointsArr.append(objpts)
                    tag_idsArr.append(detection.tag_id)

        return imgPointsArr, objPointsArr, tag_idsArr

    def _project_draw_points(
        self,
        transformation: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Projects the 3D points onto the image plane and draws those points.

        Args:
        prvecs:
            Rotation pose vector from cv:solvePnP().
        ptvecs:
            Translation pose vector from cv:solvePnP().

        Returns:
            Obtains the 3D points projected onto the image plane to be drawn.
        """

        try:
            # Project the 3D points onto the image plane
            imgpts, jac = cv.projectPoints(
                self.all_objpts,
                transformation[0],
                transformation[1],
                self.mtx,
                self.dist
            )
        except:
            raise ValueError("An error occured during projection of points.")

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

        # Need to save a copy of the previous transform
        # due to solvePnP() changing the previous before
        # variable can be used.
        unchanged_prev_transform = deepcopy(self.prev_transform)

        if imgPointsArr and objPointsArr:

            try:
                # Nx3 array
                objPointsArr = np.array(
                    objPointsArr, dtype=np.float32).reshape(-1, 3)
                # Nx2 array
                imgPointsArr = np.array(
                    imgPointsArr, dtype=np.float32).reshape(-1, 2)
            except(RuntimeError, TypeError) as error:
                raise error

            # Obtain the pose of the apriltag
            # If the last pose is None, obtain the pose with
            # no Extrinsic Guess, else use Extrinsic guess and the last pose
            try:
                if self.extrinsic_guess[0] is None or not self.enhance_ape:
                    success, pose_rvecs, pose_tvecs = cv.solvePnP(
                        objPointsArr,
                        imgPointsArr,
                        self.mtx,
                        self.dist,
                        flags=cv.SOLVEPNP_ITERATIVE
                    )
                else:
                    success, pose_rvecs, pose_tvecs = cv.solvePnP(
                        objPointsArr,
                        imgPointsArr,
                        self.mtx,
                        self.dist,
                        self.extrinsic_guess[0],
                        self.extrinsic_guess[1],
                        True,
                        flags=cv.SOLVEPNP_ITERATIVE
                    )
            except(RuntimeError, TypeError) as error:
                raise error

            transformation = (pose_rvecs, pose_tvecs)
            # TEST
            self._rmats.append(cv.Rodrigues(transformation[0])[0])
            self._tvecs.append(transformation[1])
            # TEST
            self.logger.info("Pose Obtained {}:".format(transformation))

            # If pose was found successfully
            if success:
                try:
                    mean_error = self.get_reprojection_error(
                        objPointsArr, imgPointsArr, transformation)
                    self.logger.info(
                        "Mean error: {} \n Pose rvec: {} \n Pose tvec: {}".
                        format(mean_error, pose_rvecs, pose_tvecs))
                    if mean_error < 1:
                        self.logger.info(
                            "Projecting 3D points onto the image plane.")
                        # Project the 3D points onto the image plane
                        self._project_draw_points(transformation)

                        # If this is the second frame,
                        # there would be no velocity or acc calculation
                        if self.extrinsic_guess[0] is None or not self.enhance_ape:
                            # Assign the previous pose to current pose
                            # to obtain last pose
                            # Used as an extrinisic guess
                            self.extrinsic_guess = transformation
                        else:
                            good, tran_vel, rot_vel, tran_acc, rot_acc = self.get_pose_vel_acc(
                                transformation, unchanged_prev_transform)
                            if good:
                                self.logger.info(
                                    "Obtained pose velocity and acceleration!")
                                pred_transform = self.apply_vel_acc(
                                    unchanged_prev_transform, tran_vel,
                                    tran_acc, rot_vel, rot_acc)
                                self.logger.info(
                                    "Predicted Transform {}:".format(
                                        pred_transform))

                                # Assign the previous pose to predicted pose
                                self.extrinsic_guess = pred_transform
                        # Obtain the last pose
                        # (used in the calculation for predicted pose)
                        self.prev_transform = transformation
                    else:
                        # Clear iteration if SolvePNP is 'bad'
                        self.extrinsic_guess = (None, None)
                except(RuntimeError, TypeError) as error:
                    raise error
        else:
            self.extrinsic_guess = (None, None)

    def _detect_and_get_pose(
        self, frame: np.ndarray, useflow=False, outlier_method=None, out=None
    ) -> None:
        """Obtains the pose of the dodecahedron.

        Obtains each frame from the camera,
        uses the apriltag library to detect the apriltags,
        overlays on the apriltag, and uses those detections to
        estimate the pose using OpenCV functions
        cv:solvePnP() and cv:projectPoints().

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
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Obtain AprilGroup Detected and their respective
        # tag ids, image and object points
        try:
            imgPointsArr, objPointsArr, tag_ids = self._obtain_detections(gray)
        except(RuntimeError, TypeError) as error:
            raise error

        try:
            if useflow and self._did_ape_fail(tag_ids) and self.ids_buf:
                print("Use flow and ape failed")
                imgPointsArr, objPointsArr, tag_ids, out = self._get_more_imgpts(
                    gray, imgPointsArr, objPointsArr, tag_ids,
                    outlier_method=outlier_method, out=out)
        except(RuntimeError, TypeError) as error:
            raise error

        if tag_ids:
            # Only update the queues with respective image, object points
            # and tag ids when tag ids are obtained, this allows flow
            # to be tracked for frames where no tag ids were detected.
            self._update_flow_buffers(
                gray, imgPointsArr, objPointsArr, tag_ids)

        # Using those points, estimate the pose of the dodecahedron
        self._estimate_pose(imgPointsArr, objPointsArr)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Undistorts Frame (other processing, if needed, can go here)
        """

        # Undistorts the frame
        if self.dist is not None:
            frame = self.undistort_frame(frame)

        return frame

    def overlay_camera(self, useflow, outlier_method) -> None:
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

        # Create a cv window to show images
        window = 'Camera'
        cv.namedWindow(window)

        # Open the first camera to get the video stream and the first frame
        # Change based on which webcam is being used
        # It is normally "0" for the primary webcam
        cap = cv.VideoCapture("/dev/video2")
        cap.set(3, 1280)
        cap.set(4, 720)
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
        time.sleep(2.0)
        success, frame = cap.read()

        if success:
            frame = self.process_frame(frame)
            out = frame.copy()
            # Obtains the pose of the object on the
            # frame and overlays the object.
            self._detect_and_get_pose(frame, useflow=useflow, outlier_method=outlier_method, out=out)

        while True:

            try:
                success, frame = cap.read()

                if success:
                    frame = self.process_frame(frame)
                    out = frame.copy()
                    # Obtains the pose of the object on the
                    # frame and overlays the object.
                    self._detect_and_get_pose(frame, useflow=useflow, outlier_method=outlier_method, out=out)
                else:
                    break
            except:
                continue

            # draw the text and timestamp on the frame
            cv.putText(
                frame,
                "Displaying dodecahedron with aprilgroup".format(),
                (10, 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            cv.putText(
                frame,
                datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1
            )

            # Display the object itself with points overlaid onto the object
            cv.imshow(window, self.img)
            # Display the black frame window that shows a
            # 3D drawing of the object
            cv.imshow('image', self.draw_frame)

            if useflow:
                cv.imshow('Tracker', cv.resize(
                    out, (out.shape[1]//2, out.shape[0]//2)))

            # if ESC clicked, break the loop
            if cv.waitKey(1) == 27:
                cv.destroyAllWindows()
                break
