"""
The main.py will call all the necessary functionality for this project.
So far it calibrates the camera, detects the apriltags and estimates
the pose of the aprilgroup, drawing, it onto an OpenCV window.
"""
from pathlib import Path
import argparse
from logging_results.create_logs import CustomLogger
from calibration.calibrate_camera import Calibration
from aprilgroup_pose_estimation.detect_pose import PoseDetector


def obtain_argparsers():
    """
    Creates arguments to be used during function call for
    easier testing.
    """

    # Create the parser
    parser = argparse.ArgumentParser(description="Parser used for easy testing.")

    # Add arguments

    # Arg Parsers to enhance APE by using the predicted pose,
    # if --disable-enhanced-ape, solvePnP() is used with no extrinsic guess.
    parser.add_argument('--disable-enhanced-ape', dest='enhanceape',
                        action='store_false',
                        help="Disables extrinsic guess usage to enhance APE",
                        default=True)

    parser.add_argument('--disable-opticalflow', dest='opticalflow',
                    action='store_false',
                    help="Disables optical flow and only uses APE",
                    default=True)

    parser.add_argument(
        '--outliermethod',
        type=str,
        default='opencv',
        help='If "opencv", the OpenCV outlier removal will be used, if \
        "velocity_vector, the velocity vector method will be used.')

    # Parse the arguments
    args = parser.parse_args()

    return args


def main():
    """
    Main function to create all custom loggers, and execute the detection
    and pose estimation using AprilTags.
    """

    args = obtain_argparsers()

    # Create a folder called "logs"
    log_directory = "logs"
    try:
        if not Path(log_directory).exists:
            Path.mkdir(log_directory)
    except IsADirectoryError as no_log_error:
        raise ValueError("Could not create log directory") from no_log_error

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
    det_pose = PoseDetector(det_pose_logger, mtx, dist, args.enhanceape)
    if args.opticalflow:
        det_pose.overlay_camera(args.opticalflow, args.outliermethod)
    else:
        det_pose.overlay_camera(args.opticalflow, None)


if __name__ == "__main__":
    main()
