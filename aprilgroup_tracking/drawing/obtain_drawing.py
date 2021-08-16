"""
Drawing Module
"""
import numpy as np
import cv2 as cv
import time
from drawing.plane_intersection import PlaneIntersection
from drawing.plane_points_finder import PlanePointsFinder
from drawing.dynamic_plotter import DynamicPlotter


class ObtainDrawing(PlaneIntersection):

    def __init__(self, logger, det_pose):
        """Inits PlaneCalibrator Class with a logger, camera matrix and 
        distortion coefficients, the AprilTag size used, the world coordinate
        list and AprilTag Detector Options.
        """

        self.logger = logger
        self.det_pose = det_pose
        self.world_points = self.set_world_coords()
        PlaneIntersection.__init__(self, logger, self.world_points)
        
        # self.points = []

        self.radius = 0.07
        self.pentip_position: np.ndarray = np.array([1.52068624e-01, -5.52468528e-05, -9.78557957e-02])

    def set_world_coords(self):
        """
        """
        
        obtain_plane_pts = PlanePointsFinder(self.logger, self.det_pose.mtx, self.det_pose.dist)
        if not obtain_plane_pts.try_load_world_coords():
            obtain_plane_pts.get_points()
            obtain_plane_pts.load_world_coords()
        world_coords = obtain_plane_pts.world_coords_list

        return world_coords

    
    def init_params_for_plane(self):
        """ Initialise the params for the plane.
        """
        # Plane intersection method to get the d
        # by doing this, the norm values are also initialised
        self.get_d(self.world_points[2], self.world_points[0])


    def live_drawing(self):

        self.init_params_for_plane()

        # Create a cv window to show images
        window = 'Drawing Test'
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

        # Initialise the Dynamic Graph Plotter
        m = DynamicPlotter()

        while True:

            try:
                success, frame = cap.read()
            except OSError as frame_err:
                raise OSError("Error reading the frame, \
                    please check that the webcam is connected.") from frame_err

            if success:
                frame = self.det_pose.process_frame(frame)
                # Obtains the pose of the object on the
                # frame and overlays the object.
                tag_ids, transform = self.det_pose._detect_and_get_pose(frame, True, "opencv", out=None)
                # print("type transform", type(transform))

                if not transform[0].size == 0 and not transform[1].size == 0:

                    s_c = self.det_pose.transform_marker_corners(self.pentip_position, transform).reshape(-1)
                    
                    self.logger.info("sphere center {}".format(s_c))

                    imagePoints, _ = cv.projectPoints(np.float32(self.pentip_position), transform[0], transform[1], self.det_pose.mtx, self.det_pose.dist)
                    self.logger.info("imagepts {}".format(imagePoints))
                    x = imagePoints[0,0,0]
                    y = imagePoints[0,0,1]

                    frame = cv.circle(frame, (int(x), int(y)), 5, (0,0,0), -1)

                    # pen tip trajectory
                    distance = self.shortest_distance(s_c[0], s_c[1], s_c[2])
                    self.logger.info("distance {}".format(distance))
                    is_distance_close = self.check_distance(self.radius, distance)
                    self.logger.info("is_distance_close {}".format(is_distance_close))

                    if is_distance_close and m.start:
                        m.points[0] = np.append(m.points[0], [s_c[0] * 1000])
                        m.points[1] = np.append(m.points[1], [s_c[1] * 1000])
                        m.points[2] = np.append(m.points[2], [s_c[2] * 1000])

                        # m.points[0] = np.append(m.points[0], x)
                        # m.points[1] = np.append(m.points[1], y)
                        cv.waitKey(1)

                        m.onNewData(m.points[0], m.points[1])
                        cv.waitKey(1)
                    else:
                        m.points[0] = np.append(m.points[0], np.nan)
                        m.points[1] = np.append(m.points[1], np.nan)
                        m.points[2] = np.append(m.points[2], np.nan)
                        cv.waitKey(1)
                else:
                    m.points[0] = np.append(m.points[0], np.nan)
                    m.points[1] = np.append(m.points[1], np.nan)
                    m.points[2] = np.append(m.points[2], np.nan)
                    cv.waitKey(1)

                # Display the object itself with points overlaid onto the object
                cv.imshow(window, frame)
                cv.waitKey(1)
            else:
                break
            time.sleep(.01)

            # if ESC clicked, break the loop
            if cv.waitKey(1) == 27:
            
                cv.destroyAllWindows()
                cap.release()
                cv.waitKey(1)
                # sys.exit(m.exec())