import numpy as np
import cv2
from typing import List, Dict, Tuple

class OpticalFlow:

    def __init__(self):
        self.gray_buf = []
        self.corners_buf = []
        self.objpts_buf = []
        self.ids_buf = []

        self.flow_params = dict(
                winSize = (21, 21),
                maxLevel = 3,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )

    def _did_ape_fail(self, ids):
        '''
        Returns True if aproximate pose estimation fails.
        '''

        if len(ids) <= 1:
            return  True
        else:
            return False

    def _update_flow_buffers(self, gray, corners, objpts, ids, buf_size=5):
        '''
        Adds gray, corners, and ids to their respective queues. Will remove old
        items in the buffers based on buf_size
        '''
        self.gray_buf.append(gray)
        self.corners_buf.append(corners)
        self.objpts_buf.append(objpts)
        self.ids_buf.append(ids)

        if len(self.ids_buf) > buf_size:
            self.gray_buf.pop(0)
            self.corners_buf.pop(0)
            self.objpts_buf.pop(0)
            self.ids_buf.pop(0)

    def draw_flow(self, img, p0, p1, st):
        '''
        A helper method for visualizing optical flow.
        Args:
        img:
            The image to draw on (can be color)
        p0:
            The set of marker corners from the previous frame
        p1:
            The set of marker corners from this frame
        st:
            The status return value from cv2.calcOpticalFlowPyrLK()
        '''
        
        # TODO: What happens when the valid p1 and p0 are of different shapes?
        valid_p1 = p1[st==1]
        valid_p0 = p0[st==1]
        for i in range(valid_p0.shape[0]):
            start = tuple(valid_p0[i,:].astype(np.int))
            end = tuple(valid_p1[i,:].astype(np.int))
            color = (255, 0, 255) 
            thickness = 10
            img = cv2.arrowedLine(img, start, end, color, thickness)
        return img

    def _get_p0(self, pid, buf_index):
        '''
        Indexes the buffers to retrieve the image and object points associated with a
        given marker id.
        Args:
        pid:
            int, marker id
        buf_index:
            int, the index of the buffer to use as the last frame
        '''
        prev_ids = self.ids_buf[buf_index]
        prev_corners = self.corners_buf[buf_index]
        prev_objpts = self.objpts_buf[buf_index]

        for i, j in enumerate(prev_ids):
            if j == pid:
                p_index = i
                
        p0 = np.array(prev_corners[p_index], dtype=np.float32).reshape(-1,1,2)
        objpts0 = prev_objpts[p_index]

        return p0, objpts0

    def find_more_corners(self, gray, corners, objpts, ids, out=None):
        '''
        Given a grayscale image and the corners and ids found via APE, use optical flow
        to find additional corners.
        Looks back at the buffers to see which markers were successfully detected last frame,
        but were not detected this frame. Then uses optical flow to find these markers in the
        next frame, where APE failed. If optical flow is successfull, the new corners and id are
        added to the corners and ids arrays respectively.
        
        Returns the original corners and ids arrays with the new values founded (via optical flow) added.
        Returns:
        corners - A list of 4x1x2 np.ndarray objects representing to four corners of each marker
        ids - A Nx1 list of ids of each marker. The n-th id corresponds to the corners[n] corner positions
        '''

        if ids is None:
            return None, None, None, out
        
        if len(self.gray_buf) == 0:
            print("gray buf is none")
            # cannot compute optical flow without history in the buffers
            return corners, objpts, ids, out

        if self.ids_buf[-1] is None:
            # cannot compute optical flow if the last frame contains zero found markers
            return corners, objpts, ids, out
        
        prev_ids = self.ids_buf[-1]
        print("prev ids", prev_ids)
        if ids is None:
            corners = []
            objpts = []
            ids = np.empty((0,1))
        for pid in prev_ids: # loop for all of the markers found last frame
            p_index = None
            print("pid", pid)
            print("curr ids", ids)
            if pid not in ids: # if it was not found this frame, compute flow
                print("not found in frame")
                # get previos image and object points
                p0, objpts0 = self._get_p0(pid, -1)

                # Outlier removal using abs difference between frames
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.gray_buf[-1], gray, p0, None, **self.flow_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(gray, self.gray_buf[-1], p1, None, **self.flow_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                print("d: {} \ngood: {} \n".format(d, good))

                valid_p1 = np.asarray(p1[d < 1], dtype='float32').reshape(-1, 2)
                print("valid p1", valid_p1)

                # Testing velocity vector:
                vel_vec = abs(p1-p0).reshape(-1, 2).max(-1)
                print("velocity vector: ", vel_vec)
                std_from_mean = vel_vec.mean() + 3 * vel_vec.std()
                print("3 std from mean: [", std_from_mean, "]")

                valid_p1_test = np.asarray(p1[vel_vec < std_from_mean], dtype='float32').reshape(-1, 2)
                # Initialise with trusted predictions
                p2, st2, err2 = cv2.calcOpticalFlowPyrLK(self.gray_buf[-1], gray, valid_p1_test, None, **self.flow_params)
                # Outlier removal
                vel_vec2 = abs(p2-valid_p1_test).reshape(-1, 2).max(-1)
                valid_p2_test = np.asarray(p2[vel_vec2 < vel_vec2.mean() + 3 * vel_vec2.std()], dtype='float32').reshape(-1, 2)
                print("valid p1 vel vec: ", valid_p1_test)
                print("p2: ", p2)
                print("valid p2 vel vec: ", valid_p2_test)

                if valid_p1.shape[0] == 4: # If flow was found all 4 of the marker corners. 
                    if out is not None:
                        out = self.draw_flow(out, p0, p1, st)

                    # Add the new find to the corners and ids arrays
                    corners.append(np.array(valid_p1[np.newaxis, :, :]))
                    ids.append(pid)

                    objpts.append(objpts0)

        return corners, objpts, ids, out