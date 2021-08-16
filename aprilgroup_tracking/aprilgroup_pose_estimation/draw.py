"""
Drawing Module
"""

import numpy as np
import cv2 as cv
import apriltag


class Draw():
    """Provides drawing methods for AprilTag
    detections and pose estimation.
    """

    def __init__(self, logger):
        self.logger = logger

    @staticmethod
    def draw_boxes(img: np.ndarray, imgpts: np.ndarray) -> None:
        """Draws the lines and edges on the april tag images
        to show the pose estimations.

        Args:
        self.img:
            The image containing dodecahedron with apriltags attached.
        imgpts:
            Image points of the AprilGroup returned from cv:ProjectPoints().

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

        if not np.all(imgpts):
            raise ValueError('Image points are empty.')

        # Overlay Pose onto image
        ipoints = np.round(imgpts).astype(int)
        ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]

        # Draws lines within the edges given
        for i, j in edges:
            cv.line(img, ipoints[i], ipoints[j], (0, 255, 0), 1, 16)

    @staticmethod
    def draw_2d_pts(draw_frame: np.ndarray, imgpts: np.ndarray) -> None:
        """Extracts the bounding box (x, y)-image points
        returned from cv:projectPoints() for the AprilGroup
        and convert each of the (x, y)-coordinate pairs to integers.

        Args:
        draw_frame:
            Second window to display 3D drawing of the Dodecahedron.
        imgpts:
            Image points returned from cv:projectPoints
            (mapping 3D to 2D points).

        Returns:
        Bounding box to form a 3D Dodecahedron drawing.
        """

        # Obtain the 4 points from the image points
        length = len(imgpts)
        for i in range(0, length, 4):
            (pt_a, pt_b, pt_c, pt_d) = imgpts[i:i+4].reshape(-1, 2)
            pt_b = (int(pt_b[0]), int(pt_b[1]))
            pt_c = (int(pt_c[0]), int(pt_c[1]))
            pt_d = (int(pt_d[0]), int(pt_d[1]))
            pt_a = (int(pt_a[0]), int(pt_a[1]))

            # Skip drawing if any points are out-of-frame
            max_width = draw_frame.shape[1]
            max_height = draw_frame.shape[0]
            points = [pt_a, pt_b, pt_c, pt_d]
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
            cv.line(
                draw_frame,
                pt_a, pt_b, (255, 255, 255),
                5, cv.LINE_AA)
            cv.line(
                draw_frame,
                pt_b, pt_c, (255, 255, 255),
                5, cv.LINE_AA)
            cv.line(
                draw_frame,
                pt_c, pt_d, (255, 255, 255),
                5, cv.LINE_AA)
            cv.line(
                draw_frame,
                pt_d, pt_a, (255, 255, 255),
                5, cv.LINE_AA)

    def draw_squares_and_3d_pts(self, img: np.ndarray, draw_frame: np.ndarray,
                                imgpts: np.ndarray, colour) -> None:
        """Extracts the bounding box (x, y)-image points
        returned from cv:projectPoints() for the AprilGroup
        and convert each of the (x, y)-coordinate pairs to integers.

        Args:
        img:
            Original frame data.
        draw_frame:
            Second window to display 3D drawing of the Dodecahedron.
        imgpts:
            Image points returned from cv:projectPoints
            (mapping 3D to 2D points).

        Returns:
        Bounding box to form a 3D Dodecahedron drawing,
        and image points overlay on the AprilGroup Detected.
        """

        if not np.all(imgpts):
            raise ValueError('Image points are empty.')

        # Overlay Pose onto image
        ipoints = np.round(imgpts).astype(int)
        ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]

        # Draw points obtained from cv:projectPoints()
        # overlay onto the dodecahedron object itself.
        for i in ipoints:
            if i[1] >= 0 and i[1] < 720 and i[0] >= 0 and i[0] < 1280:
                cv.circle(img, (i[0], i[1]), 5, colour, -1)

        if draw_frame is not None:
            self.draw_2d_pts(draw_frame, imgpts)

    @staticmethod
    def draw_corners(img: np.ndarray, detection: apriltag.Detection) -> None:
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
        (pt_a, pt_b, pt_c, pt_d) = detection.corners
        pt_b = (int(pt_b[0]), int(pt_b[1]))
        pt_c = (int(pt_c[0]), int(pt_c[1]))
        pt_d = (int(pt_d[0]), int(pt_d[1]))
        pt_a = (int(pt_a[0]), int(pt_a[1]))

        # Draw the bounding box of the AprilTag detection
        cv.line(img, pt_a, pt_b, (0, 255, 0), 5)
        cv.line(img, pt_b, pt_c, (0, 255, 0), 5)
        cv.line(img, pt_c, pt_d, (0, 255, 0), 5)
        cv.line(img, pt_d, pt_a, (0, 255, 0), 5)

        # Draw the center (x, y)-coordinates of the AprilTag
        (c_x, c_y) = (int(detection.center[0]), int(detection.center[1]))
        cv.circle(img, (c_x, c_y), 5, (0, 255, 255), -1)

        # Draw the tag family on the image
        tag_id = 'ID: {}'.format(detection.tag_id)
        cv.putText(img, tag_id, (pt_a[0], pt_a[1] - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    @staticmethod
    def draw_contours(draw_frame: np.ndarray, imgpts: np.ndarray) -> None:
        """Draws the contour shape of the image onto the second openCV window.

        Args:
        draw_frame:
            Second window to display 3D drawing of the Dodecahedron.
        imgpts:
            Coordinates of 3D points projected on 2D image plane.

        Returns:
        3-Dimensional shape of the image drawn on the second window.
        """

        if not np.all(imgpts):
            raise ValueError('Image points are empty.')

        # Overlay Pose onto image
        ipoints = np.round(imgpts).astype(int)
        ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]

        # Draw 3-dimensional shape of the image
        draw_shape = np.array(ipoints)
        cv.drawContours(draw_frame, [draw_shape], 0, (255, 255, 255), -1)
