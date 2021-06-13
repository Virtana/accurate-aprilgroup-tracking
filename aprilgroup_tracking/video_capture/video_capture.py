'''
TODO: To be applied separately to the class aprilgroup_pose_estimation instead of within that class...
'''

import cv2


class VideoCapture:
    def __init__(self, video_source=0):
        self.video_capture = cv2.VideoCapture(video_source)
        if not self.video_capture.isOpened():
            raise ValueError("Unable to open video source {}.".format(video_source))


    def __del__(self):
        '''
        Closes the video window if opened.
        '''
        if self.video_capture.isOpened():
            self.video_capture.release()


    def get_next_frame(self):
        '''
        Gets the frame from the video windoe.

        :return: ret: Bool that displayed that true/false status for if the video window is successfully opened.
        :return: frame: Image frame captured from the video.
        '''
        if self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return ret, frame
            else:
                return ret, None

