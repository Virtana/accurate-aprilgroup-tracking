import numpy as np
import cv2 as cv
from typing import List, Tuple


class OpticalFlow(object):
    """Tracks the flow of the dodecaPen when <=1 AprilTags are detected.

    This Class checks if APE fails, once it does it tracks the flow
    using the tag ids saved in a buffer. The tracked points are
    tested and outliers are removed. The new points that are accepted
    are then added to the image points array, with the respective object points
    and sent to the estimate pose function.

    If no tag was detected in a frame, the previous tag id is used to track
    the flow.

    Attributes:
        gray_buf: Saves gray frames.
        imgpts_buf: Saves the image points (imgpts) from AprilTag detections.
        objpts_buf: Saves the object points for the respective tag ids.
        ids_buf: Saves the tag ids detected.
        flow_params: Optical flow params used in
                    Pyramidal Lucas Kanade algorithm.
    """

    def __init__(self, logger):
        self.logger = logger
        self.gray_buf: List[object] = []
        self.imgpts_buf: List[object] = []
        self.objpts_buf: List[object] = []
        self.ids_buf: List[object] = []

        self.flow_params = dict(
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
                )

    def _did_ape_fail(self, ids) -> bool:
        """
        Returns True if aproximate pose estimation fails.
        """

        return len(ids) <= 1

    def _update_flow_buffers(
        self,
        gray,
        imgpts,
        objpts,
        ids,
        buf_size=5
    ) -> None:
        """
        Adds gray, imgpts, objpts, and ids to their respective queues.
        Will remove old items in the buffers based on buf_size.
        """

        self.gray_buf.append(gray)
        self.imgpts_buf.append(imgpts)
        self.objpts_buf.append(objpts)
        self.ids_buf.append(ids)

        if len(self.ids_buf) > buf_size:
            self.gray_buf.pop(0)
            self.imgpts_buf.pop(0)
            self.objpts_buf.pop(0)
            self.ids_buf.pop(0)

    def _draw_flow(self, img, p0, p1, st) -> np.ndarray:
        """
        A helper method for visualizing optical flow.

        Args:
        img:
            The image to draw on (can be color)
        p0:
            The set of image points from the previous frame
        p1:
            The set of image points from this frame
        st:
            The status return value from cv.calcOpticalFlowPyrLK()
        """

        # TODO: What happens when the valid p1 and p0 are of different shapes?
        valid_p1 = p1[st == 1]
        valid_p0 = p0[st == 1]
        for i in range(valid_p0.shape[0]):
            start = tuple(valid_p0[i, :].astype(np.int))
            end = tuple(valid_p1[i, :].astype(np.int))
            color = (255, 0, 255)
            thickness = 10
            img = cv.arrowedLine(img, start, end, color, thickness)
        return img

    def _get_p0(self, pid, buf_index) -> Tuple[np.ndarray, np.ndarray]:
        """
        Indexes the buffers to retrieve the image and
        object points associated with a given tag id.

        Args:
        pid:
            int, tag id
        buf_index:
            int, the index of the buffer to use as the last frame
        """

        # Retrieve the previus ids, imgpts and objpts
        prev_ids = self.ids_buf[buf_index]
        prev_imgpts = self.imgpts_buf[buf_index]
        prev_objpts = self.objpts_buf[buf_index]

        # Obtain the index for the pid
        for i, j in enumerate(prev_ids):
            if j == pid:
                pid_index = i

        # Obtain the points for the respective pid
        imgpts0 = np.array(prev_imgpts[pid_index], dtype=np.float32).reshape(-1, 1, 2)
        objpts0 = prev_objpts[pid_index]

        return imgpts0, objpts0

    def _outlier_removal(
        self,
        outlier_method,
        gray,
        imgpts0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Removes points where the tracking error or
        velocity vector is too large.
        """
        
        imgpts1, st, err = cv.calcOpticalFlowPyrLK(
            self.gray_buf[-1], gray, imgpts0, None, **self.flow_params)

        if outlier_method == "opencv":
            # Finds the difference between thr previous frame and the tracking
            # from the current frame to the previous frame.
            # If the values obtained is < 1, they are removed.

            # Outlier removal using abs difference between frames
            imgpts0r, st, err = cv.calcOpticalFlowPyrLK(
                gray, self.gray_buf[-1], imgpts1, None, **self.flow_params)

            diff = abs(imgpts0-imgpts0r).reshape(-1, 2).max(-1)
            good = diff < 1

            self.logger.info(
                "Difference: {} \n Good: {} \n".format(diff, good))
            valid_points = np.asarray(imgpts1[good], dtype='float32').reshape(-1, 2)
            self.logger.info("Valid points: {}".format(valid_points))

        elif outlier_method == "velocity_vectors":
            # Finds the difference between the tracked points and the
            # previous frame. If the values are < 3 standard deviations
            # from the mean, they are rejected. Using these trusted points
            # cv:calcOpticalFlowPyrLK() is recalled and the same outlier
            # removal is used.

            # Velocity Vector
            vel_vec = abs(imgpts1-imgpts0).reshape(-1, 2).max(-1)
            self.logger.info(
                "3 Std Dev from mean: [{}]".format(
                    vel_vec.mean() + 3 * vel_vec.std()))

            valid_first_pass = np.asarray(
                imgpts1[vel_vec < vel_vec.mean() + 3 * vel_vec.std()],
                dtype='float32').reshape(-1, 2)
            self.logger.info(
                "Valid first pass: {}".format(valid_first_pass))

            # Re-initialise with trusted predictions
            second_imgpts1, st, err = cv.calcOpticalFlowPyrLK(
                self.gray_buf[-1], gray,
                valid_first_pass, None, **self.flow_params)

            # Perform same outlier removal
            vel_vec2 = abs(second_imgpts1-valid_first_pass).reshape(-1, 2).max(-1)
            valid_points = np.asarray(
                second_imgpts1[vel_vec2 < (vel_vec2.mean() + 3 * vel_vec2.std())],
                dtype='float32').reshape(-1, 2)
            self.logger.info("Valid second pass: {}".format(valid_points))

        return valid_points, imgpts1, st

    def _get_more_imgpts(
        self,
        gray,
        imgpts,
        objpts,
        ids,
        outlier_method=None,
        out=None
    ) -> Tuple[List[object], List[object], List[object], np.ndarray]:
        """
        The grayscale image, image and object points, and tag ids
        found via APE are used in optical flow to find more image points.
        Checking the tag ids, image and object points buffers, if there
        are previous points successfully saved, the the tag ids
        that are not detected in the current frame are tracked.

        If the optical flow is successful after outlier removal, the new image
        points, respective object points and tag ids are added to their arrays
        and sent to solvePnP() to estimate a pose.

        Returns the current image points, object points
        and ids arrays with the new values added.
        """

        if ids is None:
            return None, None, None, out

        if len(self.gray_buf) == 0:
            # Cannot compute optical flow without history in the buffers
            return imgpts, objpts, ids, out

        if self.ids_buf[-1] is None:
            # Cannot compute optical flow if the last frame
            # contains zero found markers
            return imgpts, objpts, ids, out

        prev_ids = self.ids_buf[-1]
        if ids is None:
            imgpts = []
            objpts = []
            ids = np.empty((0, 1))
        for pid in prev_ids:  # Loop for all of the markers found last frame
            try:
                # If it was not found this frame, compute flow
                if pid not in ids:
                    self.logger.info("Tag ids were not found in frame.")

                    # Get previous image and object points
                    imgpts0, objpts0 = self._get_p0(pid, -1)

                    # Outlier removal using abs difference between frames
                    valid_points, imgpts1, st = self._outlier_removal(outlier_method, gray, imgpts0)

                    # If flow was found all 4 of the marker corners
                    if valid_points.shape[0] == 4:
                        if out is not None:
                            out = self._draw_flow(out, imgpts0, imgpts1, st)

                        # Add to the imgpts, objpts, and ids arrays
                        imgpts.append(np.array(valid_points[np.newaxis, :, :]))
                        ids.append(pid)
                        objpts.append(objpts0)
            except(RuntimeError, TypeError) as error:
                raise error

        return imgpts, objpts, ids, out
