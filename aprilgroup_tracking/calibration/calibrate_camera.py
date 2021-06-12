'''
Camera Calibration

Approximately > 15 images are taken of different angles and rotations of a chessboard.
These images are used to calibrate the camera.
If camera intrinsics already exists, it loads them, else it uses the images taken to initiate calibration.

Important when taking images:
1. Ensure that the chessboard is on a flat surface. You want to get the distortions caused by the camera, not the object.
2. Take many pictures as necessary. 
3. Take pictures with different angles to improve the quality of the calibration.
4. Use an environment with good illumination.

'''

import numpy as np
import cv2 as cv
import os
from pathlib import Path
import importlib.util


spec = importlib.util.spec_from_file_location("create_logs", "aprilgroup_tracking/logging/create_logs.py")
logger_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logger_module)


class Calibration:
    INTRINSIC_PARAMETERS_FILE = "aprilgroup_tracking/calibration/CameraParams.npz"
    IMAGES_FOLDER = "aprilgroup_tracking/calibration/images"


    def __init__(self, logger):
        self.logger = logger
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None

        self.chessboardSize = (9,6) # Width, Height
        self.frameSize = (1280,720) # Pixel size of Camera
        self.criteria = None
        self.objp = None
        self.objpoints = None
        self.imgpoints = None


    def try_load_intrinsic(self) -> bool:
        '''
        Load intrinsics file if it is already there.

        :return: bool: Returns True if loaded, false if not.
        '''
        try:
            self.logger.info("Trying to retrieve last intrinsic calibration parameters.")

            with np.load(self.INTRINSIC_PARAMETERS_FILE) as file:
                self.mtx, self.dist, self.rvecs, self.tvecs = [file[i] for i in ('cameraMatrix','dist','rvecs','tvecs')]
        except IOError:
            self.logger.info("Could not load previous intrinsic parameters.")
            return False
        self.logger.info("Loaded previous intrinsic parameters.")

        return True


    def start_intrinsic_calibration(self):
        '''
        Initiate calibration of the camera to obtain intrinsic values.
        '''

        # Termination Criteria
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points, e.g. (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.chessboardSize[0] * self.chessboardSize[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboardSize[0],0:self.chessboardSize[1]].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.
        self.logger.info("Starting Intrinsic calibration.")


    def save_intrinsic(self):
        ''' 
        Save Camera Parameters
        '''
        np.savez(self.INTRINSIC_PARAMETERS_FILE, cameraMatrix=self.cameraMatrix, dist=self.dist, rvecs=self.rvecs, tvecs=self.tvecs)


    def calculate_intrinsic(self):
        '''
        Calculate the intrisic values of the camera based on the chessboard size, criteria and object points.
        Save them to a .npz file.
        '''

        # Get all images in directory here.
        dirpath = self.IMAGES_FOLDER

        # Loop through all images, do oberations on images and 
        # store the image points to do calibration and undistort the images later on.
        for path, _, files in os.walk(dirpath):
            for file in files:
                filepath = Path(path) / file
                img = cv.imread(str(filepath))
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                # Find the chess board corners
                ret, corners = cv.findChessboardCorners(gray, self.chessboardSize, None)

                # If found, add object points, image points (after refining them)
                if ret == True:

                    self.objpoints.append(self.objp)
                    # Finding corners in sub pixels
                    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
                    self.imgpoints.append(corners)

                    # Draw and display the corners
                    cv.drawChessboardCorners(img, self.chessboardSize, corners2, ret)
                    cv.imshow('img', img)
                    cv.waitKey(1000) # Wait 1 sec, then go to other image.

            cv.destroyAllWindows()

        ret, self.cameraMatrix, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, self.frameSize, None, None)

        self.logger.info("Camera Calibrated: {}".format(ret))
        self.logger.info("\nCamera Matrix:\n {}".format(self.cameraMatrix))
        self.logger.info("\nDistortion Parameters:\n {}".format(self.dist))
        self.logger.info("\nRotation Vectors: \n {}".format(self.rvecs))
        self.logger.info("\nTranslation Vectors: \n {}".format(self.tvecs))

        # Save intrinsic parameters to .npz file
        self.save_intrinsic()
        self.get_reprojection_error()
        self.logger.info("Intrinsic Parameters have been saved!")


    def get_reprojection_error(self):
        '''
        Obtains the reprojection error after camera calibration. The lower the error, the better calibrated the camera.
        Values should range from ...
        '''
        # Reprojection Error
        mean_error = 0

        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.cameraMatrix, self.dist)
            error = cv.norm(self.imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error

        self.logger.info("Total error: {}".format(mean_error/len(self.objpoints)))


def main():

    # Create a folder called "logs"
    os.mkdir("logs")
    logger = logger_module.CustomLogger(log_file="logs/camera_calibration_logs", name ="camera_calibration_logs")

    calibrate = Calibration(logger)
    if calibrate.try_load_intrinsic():
        mtx, dist, rvecs, tvecs = calibrate.mtx, calibrate.dist, calibrate.rvecs, calibrate.tvecs
        print(mtx)
    else:
        calibrate.start_intrinsic_calibration()
        calibrate.calculate_intrinsic()


if __name__ == "__main__":
    main()

    


