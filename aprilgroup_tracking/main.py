'''
The main.py will call all the necessary functionality for this project.
So far it calibrates the camera, detects the apriltags and estimates the pose of the aprilgroup, drawing, it onto
an OpenCV window.
'''
import os
from logging_results.create_logs import CustomLogger
from calibration.calibrate_camera import Calibration
from aprilgroup_pose_estimation.detect_pose import DetectAndGetPose


def main():

    # Create a folder called "logs"
    log_directory = "logs"
    try:
        if not os.path.exists(log_directory):
            os.makedir(log_directory)
    except IsADirectoryError:
        raise ValueError("Could not create log directory")

    # Calibration logs
    calibration_logger = CustomLogger(log_file=log_directory+"/camera_calibration_logs", name ="camera_calibration_logs")
    # Pose Estimation Logs
    det_pose_logger = CustomLogger(log_file=log_directory+"/detection_pose_estimation_logs", name ="detection_pose_estimation_logs")

    # If Camera Parameters already exists, load them, else calculate them by calibrating the camera.
    calibrate = Calibration(calibration_logger)
    if not calibrate.try_load_intrinsic():
        calibrate.calculate_intrinsic()
        calibrate.load_intrinsic()
    mtx, dist, rvecs, tvecs = calibrate.mtx, calibrate.dist, calibrate.rvecs, calibrate.tvecs

    # Detect and Estimate Pose of the Dodecahedron
    det_pose = DetectAndGetPose(det_pose_logger, mtx, dist)
    det_pose.overlay_camera()


if __name__ == "__main__":
    main()

