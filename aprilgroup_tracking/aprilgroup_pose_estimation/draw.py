import numpy as np
import cv2
import apriltag


class Draw(object):
    """Provides drawing methods for AprilTag
    detections and pose estimation.
    """

    def __init__(self, logger):
        self.logger = logger

    def draw_boxes(self, imgpts: np.ndarray) -> None:
        """Draws the lines and edges on the april tag images
        to show the pose estimations.

        Args:
        self.img:
            The image containing dodecahedron with apriltags attached.
        self.imgpts:
            Image points of the AprilGroup returned from cv2:ProjectPoints().
        self.dist:
            Distortion Coefficients from Calibrated Camera.

        Returns:
        3D Boxes that shows the AprilGroup detected and the pose estimated.
        """

        # Bounding box for AprilTag, this will display a
        # 3D cube on detected AprilTags
        # in the pose direction
        edges = np.array([
            0, 1,
            1, 2,
            2, 3,
            3, 0,
            0, 4,
            1, 5,
            2, 6,
            3, 7,
            4, 5,
            5, 6,
            6, 7,
            7, 4
        ]).reshape(-1, 2)

        # Overlay Pose onto image
        try:
            imgpts = np.round(imgpts).astype(int)
            imgpts = [tuple(pt) for pt in imgpts.reshape(-1, 2)]
        except TypeError:
            raise TypeError

        # Draws lines within the edges given
        for i, j in edges:
            cv2.line(self.img, imgpts[i], imgpts[j], (0, 255, 0), 1, 16)

    def draw_squares_and_3d_pts(self, imgpts: np.ndarray) -> None:
        """Extracts the bounding box (x, y)-image points
        returned from cv2:projectPoints() for the AprilGroup
        and convert each of the (x, y)-coordinate pairs to integers.

        Args:
        self.img:
            Original frame data.
        self.draw_frame:
            Second window to display 3D drawing of the Dodecahedron.
        imgpts:
            Image points returned from cv2:projectPoints
            (mapping 3D to 2D points).

        Returns:
        Bounding box to form a 3D Dodecahedron drawing,
        and image points overlay on the AprilGroup Detected.
        """

        # Overlay Pose onto image
        try:
            ipoints = np.round(imgpts).astype(int)
            ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]
        except TypeError:
            raise TypeError

        # Draw points obtained from cv2:projectPoints()
        # overlay onto the dodecahedron object itself.
        try:
            for i in ipoints:
                if i[1] >= 0 and i[1] < 720 and i[0] >= 0 and i[0] < 1280:
                    cv2.circle(self.img, (i[0], i[1]), 5, (0, 0, 255), -1)
        except(RuntimeError, TypeError) as error:
            raise error

        # Obtain the 4 points from the image points
        length = len(imgpts)
        for i in range(0, length, 4):
            (ptA, ptB, ptC, ptD) = imgpts[i:i+4].reshape(-1, 2)
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))

            # Skip drawing if any points are out-of-frame
            max_width = self.draw_frame.shape[1]
            max_height = self.draw_frame.shape[0]
            points = [ptA, ptB, ptC, ptD]
            skip = False
            for point in points:
                if (point[0] >= max_width or point[0] < 0 or
                    point[1] >= max_height or point[1] < 0):
                        skip = True
                        break
            if skip:
                continue

            # Draw the 3D form of the dodecahedron from the image points
            # obtained on a second frame
            cv2.line(
                self.draw_frame,
                ptA, ptB, (255, 255, 255),
                5, cv2.LINE_AA)
            cv2.line(
                self.draw_frame,
                ptB, ptC, (255, 255, 255),
                5, cv2.LINE_AA)
            cv2.line(
                self.draw_frame,
                ptC, ptD, (255, 255, 255),
                5, cv2.LINE_AA)
            cv2.line(
                self.draw_frame,
                ptD, ptA, (255, 255, 255),
                5, cv2.LINE_AA)

    def draw_corners(self, detection: apriltag.Detection) -> None:
        """Extracts the bounding box (x, y)-coordinates for the AprilTag
        and convert each of the (x, y)-coordinate pairs to integers.

        Args:
        self.img:
            Original frame data.
        detections:
            AprilTag detections found via the AprilTag library.

        Returns:
        Bounding box with center point and tag id shown overlay
        on each AprilTag detection.
        """

        # For all detections, get the corners and draw the
        # bounding box, center and tag id

        # AprilTag Corners (Image Points)
        (ptA, ptB, ptC, ptD) = detection.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))

        # Draw the bounding box of the AprilTag detection
        cv2.line(self.img, ptA, ptB, (0, 255, 0), 5)
        cv2.line(self.img, ptB, ptC, (0, 255, 0), 5)
        cv2.line(self.img, ptC, ptD, (0, 255, 0), 5)
        cv2.line(self.img, ptD, ptA, (0, 255, 0), 5)

        # Draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(detection.center[0]), int(detection.center[1]))
        cv2.circle(self.img, (cX, cY), 5, (0, 255, 255), -1)

        # Draw the tag family on the image
        tag_id = "ID: {}".format(detection.tag_id)
        cv2.putText(self.img, tag_id, (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def draw_contours(self, imgpts: np.ndarray) -> None:
        """Draws the contour shape of the image onto the second openCV window.

        Args:
        draw_frame:
            Second window to display 3D drawing of the Dodecahedron.
        imgpts:
            Coordinates of 3D points projected on 2D image plane.

        Returns:
        3-Dimensional shape of the image drawn on the second window.
        """

        # Overlay Pose onto image
        try:
            ipoints = np.round(imgpts).astype(int)
            ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]
        except TypeError:
            raise TypeError

        # Draw 3-dimensional shape of the image
        a = np.array(ipoints)
        cv2.drawContours(self.draw_frame, [a], 0, (255, 255, 255), -1)
