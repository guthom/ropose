from typing import List
from ropose_dataset_tools.DataClasses.Dataset.Dataset import Dataset
from ropose.net.pytorch.Yolo import Yolo as Net
from ropose.net.pytorch.Util import Util, Timer
from ropose.net.pytorch.DatasetTrainingUtils import DatasetUtils
from guthoms_helpers.common_stuff.ProgressBar import ProgressBar
import ropose.pytorch_config as config

datasetUtils = DatasetUtils()

def ValidateYolo(datasets: List[Dataset], modelPath: str = None):

    net = Net(netResolution=config.yolo_InputSize)
    net.LoadPretrainedModel(modelPath)

    fps = []
    yoloDetections = []
    yoloGTs = []

    timer = Timer()

    for dataset in ProgressBar(datasets):
        rawFrame = datasetUtils.LoadRawImageToArray(dataset, crop=False, exchangeForeground=False)

        timer.Start("Yolo Detection")
        rawDetections, padding, resizeFactor = Util.PredictYolo(rawFrame, yoloNet=net)

        if rawDetections is not None:
            for i in range(0, len(rawDetections)):
                rawDetections[i] = Util.RestoreYoloBB(rawDetections[i], resizeFactor, padding)

        yoloDetections.append(rawDetections)
        yoloGTs.append(dataset.yoloData.ToYoloDetections())

        elapsed = timer.Stop(False)

        fps.append(1 / elapsed)


    return fps, yoloDetections, yoloGTs