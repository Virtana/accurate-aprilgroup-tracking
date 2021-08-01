"""
The main.py will call all the necessary functionality for this project.
So far it calibrates the camera, detects the apriltags and estimates
the pose of the aprilgroup, drawing, it onto an OpenCV window.
"""
from pathlib import Path
import argparse
from logging_results.create_logs import CustomLogger
from calibration.calibrate_camera import Calibration
from calibration.calibrate_pentip import PenTipCalibrator
from aprilgroup_pose_estimation.detect_pose import DetectAndGetPose


def obtain_argparsers():
    # Create the parser
    parser = argparse.ArgumentParser(description="Parser used for easy testing.")
    # Add arguments

    # Arg Parsers to enhance APE by using the predicted pose,
    # if --no-enhanceape, solvePnP() is used with no extrinsic guess.
    enhanceape_group = parser.add_mutually_exclusive_group(required=True)
    enhanceape_group.add_argument('--enhanceape', dest='enhanceape', action='store_true')
    enhanceape_group.add_argument('--no-enhanceape', dest='enhanceape', action='store_false')
    enhanceape_group.set_defaults(enhanceape=True)

    # Arg Parsers to determine if to use optical flow.
    opticalflow_group = parser.add_mutually_exclusive_group(required=True)
    opticalflow_group.add_argument('--opticalflow', dest='opticalflow', action='store_true')
    opticalflow_group.add_argument('--no-opticalflow', dest='opticalflow', action='store_false')
    opticalflow_group.set_defaults(opticalflow=True)

    parser.add_argument(
        '--outliermethod', 
        type=str,
        default='opencv',
        help='If "opencv", the OpenCV outlier removal will be used, if \
        "velocity_vector, the velocity vector method will be used.')
    
    parser.add_argument(
        '--calibratepentip',
        action='store_true',
        help='If used, the pen tip of the Dodecahedron will be calibrated.')

    # Parse the arguments
    args = parser.parse_args()

    return args

def main():

    args = obtain_argparsers()

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
    det_pose = DetectAndGetPose(det_pose_logger, mtx, dist, args.enhanceape, args.calibratepentip)
    if args.opticalflow:
        det_pose.overlay_camera(args.opticalflow, args.outliermethod)
    else:
        det_pose.overlay_camera(args.opticalflow, None)

    if args.calibratepentip:
        # Pen-tip Calibration Logs
        pentip_calib_logger = CustomLogger(
            log_file=log_directory+"/pen_tip_calibration_logs",
            name="pen_tip_calibration_logs")

        # Obtain the pen tip [x, y, z] sphere center
        pc = PenTipCalibrator(pentip_calib_logger, det_pose._rmats, det_pose._tvecs, det_pose)
        # Using the Algebraic Two Step Method
        fixed_tip2, base_tip2, err2 = pc._algebraic_two_step()
        # Using the Algebraic One Step Method
        fixed_tip, base_tip, err = pc._algebraic_one_step()
        pc.logger.info(
            "Algebraic Two Step \n Fixed tip: {}, \n Base tip: {} \n Error: {}".format(
                fixed_tip2, base_tip2, err2))
        pc.logger.info(
            "Algebraic One Step Fixed tip: {}, \n Base tip: {} \n Error: {}".format(
                fixed_tip, base_tip, err))


if __name__ == "__main__":
    main()
