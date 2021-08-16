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
from drawing.obtain_drawing import ObtainDrawing


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

    parser.add_argument(
        '--calibratepentip',
        action='store_true',
        help='If used, the pen tip of the Dodecahedron will be calibrated.')

    parser.add_argument('--draw', dest='draw',
                action='store_true',
                help="Allows user to draw using the AprilGroup Object.",
                default=False)

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
            print("hmm")
            Path.mkdir(log_directory)
        else:
            print("wee")
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
    # Drawing Logs
    draw_logger = CustomLogger(
        log_file=log_directory+"/drawing_logs",
        name="drawing_logs")

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
        pivot_calib = PenTipCalibrator(pentip_calib_logger, det_pose.rmats,
                                        det_pose.tvecs, det_pose)
        # Using the Algebraic Two Step Method
        fixed_tip2, base_tip2, err2 = pivot_calib.algebraic_two_step()
        # Using the Algebraic One Step Method
        fixed_tip, base_tip, err = pivot_calib.algebraic_one_step()

        # Testing the fixed tip position
        pivot_calib.test_pentip_calib_img(fixed_tip2, args.opticalflow, args.outliermethod)
        pivot_calib.test_pentip_calib_video(fixed_tip2, args.opticalflow, args.outliermethod)

    if args.draw:
        obtain_drawing = ObtainDrawing(draw_logger, det_pose)
        obtain_drawing.live_drawing()


if __name__ == "__main__":
    main()
