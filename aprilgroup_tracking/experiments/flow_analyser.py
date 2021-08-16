from pathlib import Path
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from aprilgroup_pose_estimation.pose_detector import PoseDetector


class FlowAnalyser:

    def __init__(self, logger):
        """Inits DetectAndGetPose Class with a logger,
        camera matrix and distortion coefficients, the
        two frames to be displayed, AprilTag Detector Options,
        the predefined AprilGroup Extrinsics and the AprilGroup
        Object Points.
        """

        self.logger = logger

    def chart_analysis(self, mtx, dist):

        det_pose = PoseDetector(self.logger, mtx, dist, True, False)

        experiments_dir = 'aprilgroup_tracking/experiments/test_footage'
        results_dir = 'aprilgroup_tracking/experiments/results/detections'
        try:
            if not Path(results_dir).exists:
                Path.mkdir(results_dir)
        except IsADirectoryError as no_results_dir:
            raise ValueError("Could not create log directory") from no_results_dir

        video_path = Path(experiments_dir) / 'videos'
        
        for path, _, files in os.walk(video_path):
            output = os.path.join(results_dir)

            try:
                if not Path(output).exists:
                    Path.mkdir(output)
            except IsADirectoryError as no_result_error:
                raise ValueError("Could not create result directory") from no_result_error

            self.run_experiment(path, files, det_pose, output=output)

        rname =  results_dir.split('/')[-1]
        fig = self.plot_results(results_dir)
        fname = rname + '.jpeg'
        fig.savefig(os.path.join(results_dir, fname))

    def run_experiment(self, path, footage_dir, det_pose, output=None):
        """ Run experiments for APE, ICT.
        """
        fnames = glob.glob(os.path.join(path, '*.webm'))
        for f in fnames:
            print("path weee", f)
            test_name = os.path.splitext(os.path.basename(f))[0]
            self.run_trial(f, "no_flow" + test_name, output, useflow=False, tracker=det_pose)
            self.run_trial(f, "flow" + test_name, output, useflow=True, tracker=det_pose)

    def run_trial(self, fname, test_name, output, useflow=True, tracker=None):
        """Initiate the experiements and save the results.
        """

        if tracker is None:
            raise Exception("no tracker")
        det_markers, det_frames, tot_frames = tracker.video_testing(fname, useflow=useflow, outlier_method="opencv")
        self.save_results(output, test_name, useflow, det_markers, det_frames, tot_frames)

    def save_results(self, output_dir, test_name, useflow, det_markers, det_frames, tot_frames):
        """Save experiment results.
        """
        data = {
                'test_name' : test_name,
                'useflow' : useflow,
                'det_markers' : det_markers,
                'det_frames' : det_frames,
                'tot_frames' : tot_frames
            }
        fname = os.path.join(output_dir, test_name + '_results')
        np.save(fname, data, allow_pickle=True)
            

    def plot_results(self, results_dir):
        """Plot the resuls in a bar graph.
        """

        fnames = glob.glob("aprilgroup_tracking/experiments/results/detections/*.npy")
        det_markers_flow = []
        det_markers_noflow = []
        det_frames_flow = []
        det_frames_noflow = []
        tot_frames_flow = []
        tot_frames_noflow = []

        for f in fnames:
            data = np.load(f, allow_pickle=True).item()
            if data['useflow']:
                det_markers_flow.append(data['det_markers'])
                det_frames_flow.append(data['det_frames'])
                tot_frames_flow.append(data['tot_frames'])
            else:
                det_markers_noflow.append(data['det_markers'])
                det_frames_noflow.append(data['det_frames'])
                tot_frames_noflow.append(data['tot_frames'])

        # TODO: Put blue for DPR
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        fig.tight_layout(pad=3)
        ax[0].set_title('Success At Tracking At Least One Marker Per Frame')
        ax[0].set_xlabel('Frames Per Trial')
        ax[0].set_ylabel('Number Of Frames Detected')
        ax[0].bar(tot_frames_flow, det_frames_flow, color='green', label='With Optic Flow')
        ax[0].bar(tot_frames_noflow, det_frames_noflow, color='red', label='Without Optic Flow')
        handles, labels = ax[0].get_legend_handles_labels()
        ax[0].legend(handles, labels)

        ax[1].set_title('Success At Tracking Individual Markers')
        ax[1].set_xlabel('Frames Per Trial')
        ax[1].set_ylabel('Number Of Markers Detected')
        ax[1].bar(tot_frames_flow, det_markers_flow, color='green', label='With Optic Flow')
        ax[1].bar(tot_frames_noflow, det_markers_noflow, color='red', label='Without Optic Flow')
        handles, labels = ax[1].get_legend_handles_labels()
        ax[1].legend(handles, labels)

        return fig
