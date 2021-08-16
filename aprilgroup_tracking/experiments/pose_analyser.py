import cv2 as cv
import time
from aprilgroup_pose_estimation.pose_detector import PoseDetector


class PoseAnalyser:

    def __init__(self, logger):
        """Inits DetectAndGetPose Class with a logger,
        camera matrix and distortion coefficients, the
        two frames to be displayed, AprilTag Detector Options,
        the predefined AprilGroup Extrinsics and the AprilGroup
        Object Points.
        """

        self.logger = logger

    def side_analysis(self, mtx, dist, useflow) -> None:
        """ Implements APE, ICT and DPR on the same video for live
        analysis of poses.
        """

        det_pose = PoseDetector(self.logger, mtx, dist, True, False)
        det_pose_ape = PoseDetector(self.logger, mtx, dist, True, False)

        # Open the first camera to get the video stream and the first frame
        # Change based on which webcam is being used
        # It is normally "0" for the primary webcam
        cap = cv.VideoCapture("aprilgroup_tracking/experiments/test_footage/experiments/experiment2.webm")
        cap_ape = cv.VideoCapture("aprilgroup_tracking/experiments/test_footage/experiments/experiment2.webm")

        if not cap.isOpened() and not cap_ape.isOpened():
            raise OSError("Error opening webcam, please check that the webcam \
                is connected and the correct one is referenced.")

        cap.set(3, 1280)
        cap.set(4, 720)
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
        cap_ape.set(3, 1280)
        cap_ape.set(4, 720)
        cap_ape.set(cv.CAP_PROP_AUTOFOCUS, 0)
        time.sleep(2.0)

        while True:

            try:
                success, ict_frame = cap.read()
                success_ape, ape_frame = cap_ape.read()
            except OSError as frame_err:
                raise OSError("Error reading the frame, \
                    please check that the webcam is connected.") from frame_err

            if success and success_ape:
                ict_frame = det_pose.process_frame(ict_frame)
                ape_frame = det_pose.process_frame(ape_frame)
                out = ict_frame.copy()

                # Obtains the pose of the object on the
                # frame and overlays the object.
                _, _ = det_pose._detect_and_get_pose(ict_frame, useflow=True,
                                                            outlier_method="opencv", out=out)
                _, _ = det_pose_ape._detect_and_get_pose(ape_frame, useflow=False,
                                            outlier_method=None, out=None)
            else:
                break

            # combined_images = np.concatenate((det_pose.img, det_pose_ape.img), axis=1)
            # cv.imshow('Image panel', combined_images)

            # Display the object itself with points overlaid onto the object
            cv.imshow("ICT", ict_frame)
            cv.imshow("APE", ape_frame)

            # Display the object itself with points overlaid onto the object
            if cap:
                cv.imshow("ICT", det_pose.img)
            if cap_ape:
                cv.imshow("APE", det_pose_ape.img)
            if useflow:
                cv.imshow('Tracker', cv.resize(
                    out, (out.shape[1]//3, out.shape[0]//3)))

            # if ESC clicked, break the loop
            if cv.waitKey(1) == 27:
                cv.destroyAllWindows()
                break
