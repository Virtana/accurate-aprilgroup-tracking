from experiments.flow_analyser import FlowAnalyser
from experiments.pose_analyser import PoseAnalyser


class ExperimentExecuter(FlowAnalyser, PoseAnalyser):

    def __init__(self, logger, mtx, dist, useflow):
        """Inits DetectAndGetPose Class with a logger,
        camera matrix and distortion coefficients, the
        two frames to be displayed, AprilTag Detector Options,
        the predefined AprilGroup Extrinsics and the AprilGroup
        Object Points.
        """

        self.mtx = mtx
        self.dist = dist
        self.useflow = useflow

        self.logger = logger
        FlowAnalyser.__init__(self, logger)
        PoseAnalyser.__init__(self, logger)


    def execute(self, experiment):
        if experiment == "Charts":
            self.chart_analysis(self.mtx, self.dist)
        elif experiment == "Side Analysis":
            self.side_analysis(self.mtx, self.dist, self.useflow)
