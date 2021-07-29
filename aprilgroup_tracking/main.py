"""
The main.py will call all the necessary functionality for this project.
So far it calibrates the camera, detects the apriltags and estimates
the pose of the aprilgroup, drawing, it onto an OpenCV window.
"""
import os
from logging_results.create_logs import CustomLogger
from calibration.calibrate_camera import Calibration
from calibration.calibrate_pentip import PenTipCalibrator
from aprilgroup_pose_estimation.detect_pose import DetectAndGetPose

import numpy as np
import cv2
import glob

def main():

    # Create a folder called "logs"
    log_directory = "logs"
    try:
        if not os.path.exists(log_directory):
            os.makedir(log_directory)
    except IsADirectoryError:
        raise ValueError("Could not create log directory")

    # Calibration logs
    calibration_logger = CustomLogger(
        log_file=log_directory+"/camera_calibration_logs",
        name="camera_calibration_logs")
    # Pose Estimation Logs
    det_pose_logger = CustomLogger(
        log_file=log_directory+"/detection_pose_estimation_logs",
        name="detection_pose_estimation_logs")

    # If Camera Parameters already exists,
    # load them, else calculate them by calibrating the camera.
    calibrate = Calibration(calibration_logger)
    if not calibrate.try_load_intrinsic():
        calibrate.calculate_intrinsic()
        calibrate.load_intrinsic()
    mtx, dist = calibrate.mtx, calibrate.dist

    # Detect and Estimate Pose of the Dodecahedron
    useflow = True
    det_pose = DetectAndGetPose(det_pose_logger, mtx, dist)
    det_pose.overlay_camera(useflow)

    # pc = PenTipCalibrator(det_pose._rmats, det_pose._tvecs)
    # ft, bt = pc._algebraic_two_step()
    # print("ft", ft, "bt", bt)
    
    # ft, bt = pc._algebraic_two_step()
    # ALGEBRAIC ONE STEP WITH OPTICAL FLOW
    # ft = np.array([0.11903752, -0.00930495, -0.10237714]) 
    # bt = np.array([-0.00381084, 0.05605385, 0.33691193])

    # # TEST PENTIP CALIBRATION:
    # fnames = glob.glob("aprilgroup_tracking/aprilgroup_pose_estimation/pen_tip_calibration/*.jpg")
    # for f in fnames:
    #     img = cv2.imread(f, -1)
    #     frame = det_pose.process_frame(img)
    #     out = frame.copy()
    #     transformation = det_pose._detect_and_get_pose(frame, useflow=useflow, out=out)

    #     if transformation:
    #         imagePoints, _ = cv2.projectPoints(np.float32(ft), transformation[0], transformation[1], mtx, dist)
    #         print("imagepts", imagePoints)
    #         x = imagePoints[0,0,0]
    #         y = imagePoints[0,0,1]
    #         img = cv2.circle(frame, (int(x), int(y)), 5, (0,0,0), -1)
        
    #         cv2.imshow("Tracker2", img)
    #         cv2.waitKey(0)

    # print("lengths: ", len(det_pose._tvecs), len(det_pose._rmats))
    # for (rmat, tvec) in zip(det_pose._rmats, det_pose._tvecs):
    #     print("rmat", rmat)
    #     imagePoints, _ = cv2.projectPoints(np.float32(ft), rmat, tvec, mtx, dist)
    #     print("imagepts", imagePoints)
    #     x = imagePoints[0,0,0]
    #     y = imagePoints[0,0,1]
    #     img = cv2.circle(frame, (int(x), int(y)), 5, (0,0,0), -1)
    
    #     cv2.imshow("Tracker2", img)
    #     cv2.waitKey(0)

    # cv2.destroyAllWindows()
    # cv2.waitKey(1)


if __name__ == "__main__":
    main()
