"""
The main.py will call all the necessary functionality for this project.
So far it calibrates the camera, detects the apriltags and estimates
the pose of the aprilgroup, drawing, it onto an OpenCV window.
"""
from pathlib import Path
import argparse
from logging_results.create_logs import CustomLogger
from calibration.calibrate_camera import Calibration
from aprilgroup_pose_estimation.detect_pose import DetectAndGetPose


def main():

    # Create the parser
    parser = argparse.ArgumentParser(description="Parser used for easy testing.")
    # Add arguments

    # Arg Parsers to enhance APE by using the predicted pose,
    # if --no-enhanceape, solvePnP() is used with no extrinsic guess.
    enhanceape_group = parser.add_mutually_exclusive_group(required=True)
    enhanceape_group.add_argument('--enhanceape', dest='enhanceape', action='store_true')
    enhanceape_group.add_argument('--no-enhanceape', dest='enhanceape', action='store_false')
    enhanceape_group.set_defaults(enhanceape=True)

    # Parse the arguments
    args = parser.parse_args()

    # Create a folder called "logs"
    log_directory = "logs"
    try:
        if not Path(log_directory).exists:
            Path.mkdir(log_directory)
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
    det_pose = DetectAndGetPose(det_pose_logger, mtx, dist, args.enhanceape)
    det_pose.overlay_camera()

if __name__ == "__main__":
    main()
