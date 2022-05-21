import sys, os
import torch.multiprocessing as mp
import math
from typing import Tuple
import numpy as np
from ropose.net.pytorch.RoposeNet_SimpleBaseline import RoposNet as RoposNet
from ropose.net.pytorch.Human_SimpleBaselineOriginal import HumanPoseNetOriginal as HumanNet
#from ropose.net.pytorch.Human_SimpleBaseline import HumanPoseNet as HumanNet
from ropose.net.pytorch.Yolo import Yolo
from ropose.net.pytorch.Util import Util, Timer
import ropose.pytorch_config as config
import copy
import time
import matplotlib.pyplot as plt
from ropose.scripts.pytorch.TestMethods import TestRopose, TestHumanPose, DetectYolo, DetectHumans, DetectRopose
from guthoms_helpers.input_provider.ImageDirectory import ImageDirectory
from guthoms_helpers.input_provider.VideoFile import VideoFile
from guthoms_helpers.input_provider.Camera import Camera
from guthoms_helpers.common_helpers.ColorHelper import ColorHelper
from guthoms_helpers.common_stuff.FPSTracker import FPSTracker
import ropose_dataset_tools.DataSetLoader as loader
from ropose.net.pytorch.Util import Util
from ropose_dataset_tools.DataClasses.DetectionTypes.YoloDetection import YoloDetection
import cv2

from kinematic_tracker.pose_models.PoseModelHuman17_2D import PoseModelHuman17_2D
from kinematic_tracker.pose_models.PoseModelRopose_2D import PoseModelRopose_2D

roposeTestTimer = Timer()
humanTestTimer = Timer()

def Ropose(roPoseNet, rawFrame, roposeDetections, prinTimer=False, upsampleOutput=True,
           upsamplingOriginalSize=False):
    roposeTestTimer.Start()
    roposeRet = ([], [], [], [], [], [])

    if roposeDetections.__len__() > 0:
        roposeRet = DetectRopose(roPoseNet, rawFrame, roposeDetections, upsampleOutput=upsampleOutput,
                                 upsamplingOriginalSize=upsamplingOriginalSize, printTimer=prinTimer)

    elapsed = roposeTestTimer.Stop(prinTimer)
    return roposeRet

def HumanPose(humanNet, rawFrame, humanDetections, prinTimer=False, upsampleOutput=False, upsamplingOriginalSize=False):
    humanTestTimer.Start()
    humanRet = ([], [], [], [], [], [])

    if humanDetections.__len__() > 0:
        humanRet = DetectHumans(humanNet, rawFrame, humanDetections, upsampleOutput=upsampleOutput,
                                printTimer=prinTimer)

    elapsed = humanTestTimer.Stop(prinTimer)
    return humanRet

def BGRToRGB(image: np.array, targetSize: Tuple[int, int] = (1280, 720)) -> np.array:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, targetSize)

    return image

mp.set_start_method('spawn')

directory = config.roposeFineTuneDatasets[0]+"/depthcam1/rgb0"

inputProvider = ImageDirectory(dirPath=directory, sort=True, loop=False, useBuffer=False)

yolo = Yolo(netResolution=config.yolo_InputSize)
yolo.LoadPretrainedModel(config.roPoseYoloWeights)

roPoseNet = RoposNet(netResolution=config.inputRes)
roPoseNet.ShareMemory()
roPoseNet.LoadPretrainedModel(config.roPoseNetWeights)

humanNet = HumanNet(origWeightsPath=config.originalHumanPoseModelPath, netResolution=config.inputRes)
humanNet.ShareMemory()
fpsTracker = FPSTracker(meanCount=20)

counter = 0

'''
display = cv2.namedWindow('image_raw_estimated', cv2.WINDOW_NORMAL)
display2 = cv2.namedWindow('rawYolo', cv2.WINDOW_NORMAL)
display3 = cv2.namedWindow('TrackingHelp', cv2.WINDOW_NORMAL)
display4 = cv2.namedWindow('image_tracked', cv2.WINDOW_NORMAL)
'''