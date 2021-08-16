"""
Module to calibrate pen-tip,
"""

import time
from typing import Tuple
import glob
import numpy as np
import cv2 as cv
from aprilgroup_pose_estimation.pose_detector import PoseDetector


class PenTipCalibrator(PoseDetector):
    """Estimates the pen tip position c and sphere center c'.

    To calibrate the pen tip position, the object poses obtained from the DodecaPen
    rotating around a fixed point is needed.
    Note that it is important for the pen tip to be fixed on a surface when obtaining 
    these poses.
    
    This Class takes the rotational matrices and translational vectors obtained
    from moving the DodecaPen at a fixed point and obtains the
    least squares estimate of the pen tip position.

    With a fixed tip point Ft, given the poses where tip is also
    fixed with respect to the robot base frame, B, we do not have Bt.
    Then for all i,j poses, we have Rmats * Ft + tvecs = Bt
    Formulating transform = [Ri, ti] and casting the equation as Ax=b,
    we can obtain a stacked system of equations for some
    set of indices {(i,j)}_i!=j:
    [Ri - Rj]      [tj - ti]
    [  ...  ] Ft = [  ...  ]
    [  ...  ]      [  ...  ]

    We then isolate Ft and estimate Bt as the average over
    {Rmat_i * Ft + tvec_i}_i.

    Attributes:
        rmats: Rotation matrices obtained from passing
                rvecs into cv2:Rodrigues.
        tvecs: Translational Vectors obtained from cv2:solvePnP()
    """

    def __init__(self, logger, rmats, tvecs, det_pose: PoseDetector):
        self.logger = logger
        if not type(det_pose) == PoseDetector:
            raise ValueError('Parameter det_pose must be of type DetectAndGetPose.')
        self.det_pose = det_pose
        if not rmats and not tvecs:
            raise ValueError("The rotation matrices and translation vectors must be supplied.")
        self._rmats = rmats
        self._tvecs = tvecs

        self.logger.info("length of tvecs: {}".format(len(self._tvecs)))
        self.logger.info("length of rmats: {}".format(len(self._rmats)))

    def algebraic_two_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """Obtains the fixed and base tip points from the DodecaPen
        using the algebraic two step algorithm as found here:
        https://www.yanivresearch.info/writtenMaterial/pivotCalib-SPIE2015.pdf
        """

        # [Ri - Ri+1] for all i in rmats
        A = np.vstack([self._rmats[i] - self._rmats[i+1]
                    for i in range(len(self._rmats)-1)])

        # [ti+1 - ti] for all i in tvecs
        dbfps = [(self._tvecs[i+1].T - self._tvecs[i].T).reshape((3, 1))
                for i in range(len(self._tvecs)-1)]
        b = np.vstack(dbfps)

        # Linear least squares estimate of A and b
        fixed_tip = np.linalg.lstsq(A, b, rcond=None)[0].reshape(-1)

        # {Ri @ x + ti}
        rmat_c_tvecs = []
        for (rmat, tvec) in zip(self._rmats, self._tvecs):
            rmat_c_tvecs.append(((rmat @ fixed_tip) + tvec.T).reshape(-1))

        # Average of all {Ri @ x + ti}
        sphere_center = np.average(rmat_c_tvecs, axis=0)

        # Calculate the 3D residual RMS error
        residual_vectors = np.array((
            np.array(rmat_c_tvecs)).reshape(len(self._tvecs), 3))
        residual_norms = np.linalg.norm(residual_vectors, axis=1)
        residual_rms = np.sqrt(np.mean(residual_norms ** 2))

        self.logger.info(
            "Algebraic Two Step \n Fixed tip: {}, \n Base tip: {} \n Error: {}".format(
                fixed_tip, sphere_center, residual_rms))

        return fixed_tip, sphere_center, residual_rms

    def algebraic_one_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """Obtains the fixed and base tip points from the DodecaPen
        using the algebraic one step algorithm as found here:
        https://www.yanivresearch.info/writtenMaterial/pivotCalib-SPIE2015.pdf
        """

        # Create arrays which match the dims and shapes
        # expected by linalg.lstsq
        t_vecs = []
        for i in range(len(self._tvecs)-1):
            t_vecs.append(self._tvecs[i].T)
        t_i_shaped = np.array(self._tvecs).reshape(-1,1)

        r_i_shaped = []
        for r in self._rmats:
            r_i_shaped.extend(np.concatenate((r, -np.identity(3)), axis=1))
        r_i_shaped = np.stack(r_i_shaped)

        # Run least-squares, extract the positions
        lstsq = np.linalg.lstsq(r_i_shaped, -t_i_shaped, rcond=None)
        fixed_tip = lstsq[0][0:3].reshape(-1)
        sphere_center = lstsq[0][3:6].reshape(-1)

        # Calculate the 3D residual RMS error
        residual_vectors = np.array((
            t_i_shaped + r_i_shaped@lstsq[0]).reshape(len(self._tvecs), 3))
        residual_norms = np.linalg.norm(residual_vectors, axis=1)
        residual_rms = np.sqrt(np.mean(residual_norms ** 2))

        self.logger.info(
            "Algebraic One Step \n Fixed tip: {}, \n Base tip: {} \n Error: {}".format(
                fixed_tip, sphere_center, residual_rms))

        return fixed_tip, sphere_center, residual_rms

    def test_pentip_calib_img(self, fixed_tip, check_opt_flow, outliermethod):
        """Tests the fixed point on images using transforms obtained
        from the DetectAndGetPose class. The fixed point is projected
        onto the images using cv:ProjectPoints()."""

        # Path to images for testing the pen tip position
        frames = glob.glob("aprilgroup_tracking/aprilgroup_pose_estimation/pen_tip_calib_usb/test/*.jpg")

        for f in frames:
            frame = cv.imread(f, -1)
            frame = self.det_pose.undistort_frame(frame, self.det_pose.mtx, self.det_pose.dist)
            transform = self.det_pose._detect_and_get_pose(frame, check_opt_flow, outliermethod, out=None)

            if not transform[0].size == 0 and not transform[1].size == 0:
                imagePoints, _ = cv.projectPoints(np.float32(fixed_tip), transform[0], transform[1], self.det_pose.mtx, self.det_pose.dist)
                print("imagepts", imagePoints)
                x = imagePoints[0,0,0]
                y = imagePoints[0,0,1]
                frame = cv.circle(frame, (int(x), int(y)), 5, (0,0,0), -1)
            cv.imshow("Pen Tip Calibration Test", frame)
            cv.waitKey(0)

        cv.destroyAllWindows()
        cv.waitKey(1)

    def test_pentip_calib_video(self, fixed_tip, check_opt_flow, outliermethod):
        """Tests the fixed point on images using transforms obtained
        from the DetectAndGetPose class. The fixed point is projected
        onto the images using cv:ProjectPoints()."""

        # Create a cv window to show images
        window = 'Pen Tip Calibration Test'
        cv.namedWindow(window)

        # Open the first camera to get the video stream and the first frame
        # Change based on which webcam is being used
        # It is normally "0" for the primary webcam
        cap = cv.VideoCapture("/dev/video2", cv.CAP_V4L2)

        if not cap.isOpened():
            raise OSError("Error opening webcam, please check that the webcam \
                is connected and the correct one is referenced.")

        cap.set(3, 1280)
        cap.set(4, 720)
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
        time.sleep(2.0)

        while (cap.isOpened()):

            try:
                success, frame = cap.read()
            except OSError as frame_err:
                raise OSError("Error reading the frame, \
                    please check that the webcam is connected.") from frame_err

            if success:
                frame = self.det_pose.undistort_frame(frame, self.det_pose.mtx, self.det_pose.dist)
                # Obtains the pose of the object on the
                # frame and overlays the object.
                transform = self.det_pose._detect_and_get_pose(frame, check_opt_flow, outliermethod, out=None)

                if not transform[0].size == 0 and not transform[1].size == 0:
                    imagePoints, _ = cv.projectPoints(np.float32(fixed_tip), transform[0], transform[1], self.det_pose.mtx, self.det_pose.dist)
                    print("imagepts", imagePoints)
                    x = imagePoints[0,0,0]
                    y = imagePoints[0,0,1]
                    frame = cv.circle(frame, (int(x), int(y)), 5, (0,0,0), -1)

                # Display the object itself with points overlaid onto the object
                cv.imshow(window, frame)
            
            else:
                break

            # if ESC clicked, break the loop
            if cv.waitKey(1) == 27:
                cv.destroyAllWindows()
                break
