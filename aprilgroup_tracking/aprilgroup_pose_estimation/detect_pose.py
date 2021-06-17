'''
Detects and Estimates the Pose of the AprilGroup attached to a Dodecahedron.
'''

import json
import numpy as np
import cv2
import apriltag
from pathlib import Path
import datetime


class DetectAndGetPose:
    DIRPATH = 'aprilgroup_tracking/aprilgroup_pose_estimation'
    JSON_FILE = 'april_group.json'
    

    def __init__(self, logger, mtx, dist):
        self.logger = logger
        self.mtx = mtx              # Camera matrix
        self.dist = dist            # Camera distortions
        self.tvecs = None           # Object translation vectors
        self.rvecs = None           # Object rotation vectors
        self.markersize = None      # Apriltag markers size
        self.markercorners = None   # Corners of the apriltags in the world frame
        self.opoints = None         # Object points
        self.imgpts = None          # 3D points obtained from cv2:projectPoints
        self.mrv = None             # Object rotation matrix from Rodrigues
        self.extrinsics = None      # Tag size, translation and rotation vectors for predefined aprilgroup
        self.img = None             # Original Image
        self.overlay = None         # Overlay Image


    # Append multiple value to a key in dictionary
    def add_values_in_dict(self, sample_dict, key, list_of_values):
        '''
        Append multiple values to a key in the given dictionary

        :param: sample_dict: Dictionary to add the values to
        :param: key: The key in the dictionary
        :param: list_of_values: The values to be assigned to a specific key

        :return: sample_dict: The dictionary with the values added to their specific key
        '''
        if key not in sample_dict:
            sample_dict[key] = list()
        sample_dict[key].extend(list_of_values)
        return sample_dict

    
    def get_extrinsics(self):
        '''
        Obtain the tag sizes, rvecs and tvecs for each apriltag from the .json file

        :param: json_file: The json file to be used to obtain the extrinsics
        :return: extrinsics: A dict with all the extrinsic values paired to their corresponding tag_id
        '''

        # Extrinsics Dict
        extrinsics = {}

        # Opening json file containing the dodecahedron extrinsics
        filepath = Path(self.DIRPATH) / self.JSON_FILE
        with open (filepath,"r") as f:
            data = json.load(f)

        for key, tags in data['tags'].items():
            # Size of the tags
            tag_sizes = tags['size']
            # Turning tvec into N x 1 array
            tvecs = np.array(tags['extrinsics'][:3], dtype=np.float32).reshape((3,1))
            # Turning rvec into N x 1 array
            rvecs = np.array(tags['extrinsics'][-3:], dtype=np.float32).reshape((3,1))
            # Add extrinsics to their specific tag_id
            extrinsics = self.add_values_in_dict(extrinsics, int(key), [tag_sizes, tvecs, rvecs])
    
        # Closing file
        f.close()

        return extrinsics


    def undistort_frame(self, frame):
        '''
        Undistorts the camera frame given the camera matrix and distortion values from camera calibration

        :param: frame: Current camera frame
        :param: self.mtx: Camera Matrix
        :param: self.dist: Distortion Coefficients from Calibrated Camera

        :return: dst: Undistorted frame
        '''
        # Height and Width of the camera frame
        h,  w = frame.shape[:2]

        # Get the camera matrix and distortion values
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))

        # Undistort Frame
        dst = cv2.undistort(frame, self.mtx, self.dist, None, newCameraMatrix)

        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        return dst


    def draw_boxes(self, edges):
        '''
        Draws the lines and edges on the april tag images
        to show the pose estimations.

        :param: self.img: the image containing dodecahedron with apriltags attached 
        :param: self.imgpts: corner of apriltags
        :param: edges: edges of apriltags
        :output: The boxes that show the pose estimation
        '''

        # Overlay Pose onto image
        self.imgpts = np.round(self.imgpts).astype(int)
        self.imgpts = [tuple(pt) for pt in self.imgpts.reshape(-1, 2)]

        # Draws lines within the edges given
        for i, j in edges:
            cv2.line(self.img, self.imgpts[i], self.imgpts[j], (0, 255, 0), 1, 16)


    def draw_squares(self, detections):
        '''
        Extract the bounding box (x, y)-coordinates for the AprilTag
        and convert each of the (x, y)-coordinate pairs to integers

        :param: img: Original picture data
        :param: detections: AprilTag detections found via the AprilTag library

        :output: Bounding box with center point and tag id shown overlay on each AprilTag detection.
        '''

        # For all detections, get the corners and draw the bounding box, center and tag id
        for detection in detections:

            # AprilTag Corners (Image Points)
            (ptA, ptB, ptC, ptD) = detection.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))

            # Draw the bounding box of the AprilTag detection
            cv2.line(self.img, ptA, ptB, (0, 255, 0), 2)
            cv2.line(self.img, ptB, ptC, (0, 255, 0), 2)
            cv2.line(self.img, ptC, ptD, (0, 255, 0), 2)
            cv2.line(self.img, ptD, ptA, (0, 255, 0), 2)

            # Draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(detection.center[0]), int(detection.center[1]))
            cv2.circle(self.img, (cX, cY), 5, (0, 0, 255), -1)

            # Draw the tag family on the image
            tag_id = "ID: {}".format(detection.tag_id)
            cv2.putText(self.img, tag_id, (ptA[0], ptA[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    def draw_contours(self):
        '''
        Draws the shape of the image onto the second openCV window

        :param: self.overlay: Overlay picture data 
        :param: self.imgpts: Coordinates of 3D points projected on 2D image plane

        :output: 3-Dimensional shape of the image drawn on the second window.
        '''

        # Overlay Pose onto image
        ipoints = np.round(self.imgpts).astype(int)
        ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]

        # Draw 3-dimensional shape of the image
        a = np.array(ipoints)
        cv2.drawContours(self.overlay, [a], 0, (255,255,255), -1)


    def rotate_marker_corners(self):
        '''
        Rotates and Translates the apriltag by the rotation and translation vectors given.

        :param: self.markersize: Size of the apriltag
        :param: self.rvec: Rotation vector of the apriltag
        :param: self.tvec: Translation vector of the apriltag

        :output: self.opoints: Object points to be used with cv2:projectPoints()
        :output: self.markercorners: Corners of Apriltag after rotation and translation has been applied,
                these are the 3D points to be supplied to solvePnP() to obtain the pose of the apriltag.
        :output: self.mrv: Rotation Matrix obtained from Rodrigues
        '''

        self.markercorners = [] # Apriltag corners after rotation and translation
        self.opoints = [] # Object points for the apriltag

        # Apriltag radius
        mhalf = self.markersize / 2.0

        # Convert rotation vector to rotation matrix: markerworld -> cam-world
        self.mrv, jacobian = cv2.Rodrigues(self.rvecs)

        # In the marker world, the corners are all in the xy-plane, so z is zero at first
        X = mhalf * self.mrv[:,0] # Rotate the x = mhalf
        Y = mhalf * self.mrv[:,1] # Rotate the y = mhalf
        minusX = X * (-1)
        minusY = Y * (-1)

        # Move the point from sq center of the frame and apply the transform and rotation to get the object movement
        # Calculate 4 corners of the apriltag in the camera world. The corners are enumerated clockwise
        self.markercorners.append(np.add(minusX, Y)) # Upper left in marker world
        self.markercorners.append(np.add(X, Y)) # Upper right in marker world
        self.markercorners.append(np.add( X, minusY)) # Lower right in marker world
        self.markercorners.append(np.add(minusX, minusY)) # Lower left in marker world

        # If tvec given, move all points by tvec
        if self.tvecs is not None:
            C = self.tvecs.reshape(-1, 3) # Center of apriltag in camera world
            for i, mc in enumerate(self.markercorners):
                self.markercorners[i] = np.add(C,mc) # Add tvec to each corner
        
        # Type needed when used as input to cv2:solvePnp() and cv2:projectPoints()
        self.markercorners = np.array(self.markercorners,dtype=np.float32).reshape(-1, 3) 

        # Calculating the object points to be supplied to cv2:projectPoints()
        self.opoints = np.copy(self.markercorners)
        self.opoints[:, 2] =  -2*mhalf
        self.opoints = np.vstack((self.markercorners, self.opoints))
        

    def project_draw_points(self, prvecs, ptvecs):
        '''
        Projects the 3D points onto the image plane and
        draws those points.

        :param: prvecs: Rotation pose vector from cv2:solvePnP()
        :param: ptvecs: Translation pose vector from cv2:solvePnP()

        :output: self.imgpts: Obtains the 3D points onto the image plane to be drawn in estimate_pose()
        '''

        # Object points array to be supplied to cv2:projectPoints()
        opointsArr = []

        # Obtain the object points of all apriltags that form an aprilgroup 
        for i in self.extrinsics:
            self.markersize = self.extrinsics[i][0] # Size of the AprilTag markers
            self.rvecs = self.extrinsics[i][2] # Rotation Vector of AprilTag
            self.tvecs = self.extrinsics[i][1] # Translation Vector of AprilTag
            self.rotate_marker_corners() # Rotates and Translates the AprilGroup
            opointsArr.append(self.opoints)

        opointsArr = np.array(opointsArr).reshape(-1, 3) # Shape needed for cv2:projectPoints()

        # Project the 3D points onto the image plane
        self.imgpts, jac = cv2.projectPoints(opointsArr, prvecs, ptvecs, self.mtx, self.dist)


    def estimate_pose(self, detection_results):
        '''
        Gets the pose of the object with apriltags attached.

        :param: self.mtx: Camera Matrix
        :param: detection_results: AprilTags object detected

        :output: self.draw_contours: The pose of obtained and the contours found from cv2:projectPoints() are drawn.
        '''
        # If the camera was calibrated and the matrix is supplied
        if self.mtx is not None:
            # Image points are the corners of the apriltag
            imagePoints = detection_results[0].corners.reshape(1,4,2) 

            # Draw square on all AprilTag edges
            self.draw_squares(detection_results)

            # Obtain the extrinsics from the .json file for the first apriltag detected
            self.markersize = self.extrinsics[detection_results[0].tag_id][0] # Size of the AprilTag markers
            self.rvecs = self.extrinsics[detection_results[0].tag_id][2] # Rotation Vector of AprilTag
            self.tvecs = self.extrinsics[detection_results[0].tag_id][1] # Translation Vector of AprilTag

            # Obtain the object points (marker_corners) of the apriltag detected
            self.rotate_marker_corners() # Rotates and Translates the AprilGroup

            # Obtain the pose of the apriltag
            success, prvecs, ptvecs = cv2.solvePnP(self.markercorners, imagePoints, self.mtx, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)

            # If pose was found successfully
            if success:
                self.logger.info("Projecting the 3D points onto the image plane...")
                # Project the 3D points onto the image plane
                self.project_draw_points(prvecs, ptvecs)

                self.logger.info("Drawing the points...")
                # Draw the contours of the image
                self.draw_contours()


    def detect_and_get_pose(self, frame):  
        '''
        This function takes each frame from the camera, 
        uses the apriltag library to detect the apriltags, overlays on the apriltag,
        and uses those detections to estimate the pose using OpenCV functions
        solvePnP() and projectPoints().

        :param: frame: Each frame from the camera
        :output: Pose of the AprilGroup overlayed on the dodecapen with apriltags attached object and the object itself
        '''

        # Get the frame from the Video
        self.img = frame

        # Apply grayscale to the frame to get proper AprilTag Detections
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get all extrinsics and tag sizes from the .json file
        self.extrinsics = self.get_extrinsics()

        # AprilTag detector options
        options = apriltag.DetectorOptions(families='tag36h11',
                                        border=1,
                                        nthreads=4,
                                        quad_decimate=1.0,
                                        quad_blur=0.0,
                                        refine_edges=True,
                                        refine_decode=False,
                                        refine_pose=True,
                                        debug=False,
                                        quad_contours=True)

        detector = apriltag.Detector(options)

        # Detect the apriltags in the image
        detection_results, dimg = detector.detect(gray, return_image=True)

        # Amount of april tags detected
        num_detections = len(detection_results)
        self.logger.info('Detected {} tags.\n'.format(num_detections))

        # Overlay on the dodecahedron with AprilTags
        self.overlay = self.img // 2 

        # If 1 or more apriltags are detected, estimate and draw the pose
        if num_detections > 0:
            self.estimate_pose(detection_results)


    def process_frame(self, frame):
        '''
        Undistorts Frame (other processing, if needed, can go here)
        :return: frame: Processed Frame
        '''

        # Undistorts the frame
        if self.dist is not None:
            frame = self.undistort_frame(frame)
        
        return frame


    def overlay_camera(self):
        '''
        This function creates a new camera window, to show both the pose estimation boxes
        drawn on the apriltags, and projections overlayed onto the incoming frames. 
        This will allow us to quickly see the limitations of the baseline approach, i.e. when detection fails.

        :output: Two video captured windows, one displays the object and the pose overlaid over the object, the
        other displays the drawing of the object pose.
        '''

        # Create a cv2 window to show images
        window = 'Camera'
        cv2.namedWindow(window)

        # Open the first camera to get the video stream and the first frame
        cap = cv2.VideoCapture(0)
        success, frame = cap.read()

        if success:
            frame = self.process_frame(frame)
            # Obtains the pose of the object on the frame and overlays the object.
            self.detect_and_get_pose(frame)

        while True:

            # Get previous frame
            prev_frame = frame[:]

            success, frame = cap.read()

            if success:
                frame = self.process_frame(frame)
                # Obtains the pose of the object on the frame and overlays the object.
                self.detect_and_get_pose(frame)
            else:
                break

            # draw the text and timestamp on the frame
            cv2.putText(frame, "Displaying dodecahedron with aprilgroup".format(), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # Display the object itself with points overlaid onto the object
            cv2.imshow(window, self.img)
            # Display the overlay window that shows a drawing of the object
            cv2.imshow('image', self.overlay)
        
            # if ESC clicked, break the loop
            if  cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                break


