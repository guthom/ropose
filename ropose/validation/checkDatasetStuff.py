import sys, os
import numpy as np
from ropose.net.pytorch.RoposeNet_SimpleBaseline import RoposNet as Net
from ropose.net.pytorch.Util import Util, Timer
import ropose.pytorch_config as config
from guthoms_helpers.base_types.BoundingBox2D import BoundingBox2D
from guthoms_helpers.base_types.Pose2D import Pose2D
from ropose_dataset_tools.DataOrganizer import DataOrganizer
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ropose_dataset_tools.DataClasses import Dataset
import ropose_dataset_tools.DataSetLoader as loader
from ropose.scripts.pytorch.TestMethods import TestRopose, DetectRopose
from ropose.validation.Validator import Validator
import numpy as np
from typing import List
from guthoms_helpers.filesystem.DirectoryHelper import DirectoryHelper
from guthoms_helpers.filesystem.FileHelper import FileHelper
from guthoms_helpers.common_stuff.ProgressBar import ProgressBar

evaluationDataBasePath = os.path.join(config.evalPath, "RoPose_Keypoints")
DirectoryHelper.CreateIfNotExist(evaluationDataBasePath)

datasets = []
datasets.extend(loader.LoadDir(path=config.roposeEvalDataPath))
datasets.extend(loader.LoadDir(path=config.roposeTestDataPath))
datasets.extend(loader.LoadDir(path=config.realDataPath))

baseSegmentLengts = []
boundingBoxDiags = []
boundingBoxHeights = []
boundingBoxWidths = []

for dataset in ProgressBar(datasets):
    bb: BoundingBox2D = dataset.rgbFrame.boundingBox
    boundingBoxDiags.append(bb.DiagLength())
    boundingBoxHeights.append(bb.height)
    boundingBoxWidths.append(bb.width)

    poses: List[Pose2D] = dataset.rgbFrame.resizedReprojectedPoints
    baseSegmentLengts.append(poses[0].trans.Distance(poses[1].trans))

print("baseSegmentLengts: " + str(np.mean(baseSegmentLengts)) + "/" + str(np.max(baseSegmentLengts)) + "/"
                                  + str(np.min(baseSegmentLengts)))
print("boundingBoxDiags: " + str(np.mean(boundingBoxDiags)) + "/" + str(np.max(boundingBoxDiags)) + "/"
                                  + str(np.min(boundingBoxDiags)))
print("boundingBoxHeights: " + str(np.mean(boundingBoxHeights)) + "/" + str(np.max(boundingBoxHeights)) + "/"
                                  + str(np.min(boundingBoxHeights)))
print("boundingBoxWidths: " + str(np.mean(boundingBoxWidths)) + "/" + str(np.max(boundingBoxWidths)) + "/"
                                  + str(np.min(boundingBoxWidths)))