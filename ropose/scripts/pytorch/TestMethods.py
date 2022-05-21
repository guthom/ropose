import numpy as np
from ropose.net.pytorch.Util import Util, Timer
import ropose.pytorch_config as config
from typing import List, Tuple
from ropose.net.pytorch.DatasetTrainingUtils import DatasetUtils
import copy
from ropose_dataset_tools.DataClasses.Dataset.BoundingBox import BoundingBox
from ropose_dataset_tools.DataClasses.DetectionTypes.YoloDetection import YoloDetection
from ropose_dataset_tools.DataClasses.DetectionTypes.KeypointDetection import KeypointDetection
from ropose.net.pytorch.Human_SimpleBaselineOriginal import HumanPoseNetOriginal
from guthoms_helpers.base_types.Pose2D import Pose2D
import torch
import cv2

printTestTimers = True
datsetUtils = DatasetUtils(useGreenscreeners=False)
def TestAbstract(net, dataset,  rejectionTH, upsampleOutput: bool, upsamplingMethod: str = "bilinear",
                 upsamplingOriginalSize: bool = False,  printTimer=False):
    timerAll = Timer()
    timer = Timer()

    timer.Start("loading")
    datsetUtils.LoadXY(dataset)

    rawFrame = datsetUtils.LoadRawImageToArray(dataset, crop=False)

    frames = []
    paddings = []
    resizeFactors = []
    boundingBoxes = []
    gtKeypoints = []
    originalSizes = []

    for boundingBox in dataset.yoloData.boundingBoxes:
        rawYoloFrame = boundingBox.CropImage(rawFrame)
        x, padding, paddedResizeFactor = Util.PrepareRawInput(rawYoloFrame, config.inputRes)
        originalSizes.append((int(boundingBox.width), int(boundingBox.height)))
        resizeFactors.append(paddedResizeFactor)
        paddings.append(padding)
        boundingBoxes.append(boundingBox)

        x = Util.ToFloat64Image(x)
        x = x.transpose((2, 0, 1))
        x = net.PreprocessInput(torch.from_numpy(x).float())
        gtKeypoints.append(dataset.yoloData.keypoints)
        frames.append(x)

    # stack yolo batches together
    frames = torch.stack(frames)

    timerAll.Start("Processing All")
    loadingTime = timer.Stop(printTimer)

    timer.Start("Prediction")
    pred = net.Predict(frames)
    timer.Stop(printTimer)

    #Util.ShowHeatmaps(pred[0, :, :, :])

    dim = pred.shape

    timer.Start("ExtractKeypoints")
    #raise Exception("Debug first !!!!")
    if not upsampleOutput:
        keypoints, probs = Util.ExtractKeypoints(pred[:, 0:dim[1]-1, :, :])
        keypoints = Util.RestoreKeyPoints(keypoints, resizeFactor=resizeFactors, paddings=paddings,
                                          boundingBoxes=boundingBoxes)
        keypoints = Util.FilterDetection(keypoints, probs, rejectionTH)
    else:
        fakeTargetSizes = [config.inputRes for i in range(0, dim[0])]
        if not upsamplingOriginalSize:
            upsamepledHeatmaps = Util.UpsampleBatchHeatmapsGPU(heatmaps=pred,
                                                               targetSizes=fakeTargetSizes,
                                                               method=upsamplingMethod)
        else:
            upsamepledHeatmaps = Util.UpsampleBatchHeatmapsGPU(heatmaps=pred,
                                                               targetSizes=fakeTargetSizes,
                                                               resizeFactors=resizeFactors,
                                                               method=upsamplingMethod)

        keypoints = []
        probs = []

        for i in range(0, upsamepledHeatmaps.__len__()):
            subKeypoints, subProbs = Util.ExtractKeypoints(torch.unsqueeze(upsamepledHeatmaps[i][0:dim[1] - 1, :, :],
                                                                           dim=0))
            keypoints.append(subKeypoints[0])
            probs.append(subProbs[0])

        if not upsamplingOriginalSize:
            keypoints = Util.RestoreKeyPoints(keypoints, resizeFactor=resizeFactors,
                                              outputSize=config.inputRes,
                                              paddings=paddings,
                                              boundingBoxes=boundingBoxes)
        else:
            keypoints = Util.RestoreKeyPointsUpsampled(keypoints, paddings=paddings, boundingBoxes=boundingBoxes)

        keypoints = Util.FilterDetection(keypoints, probs, rejectionTH)


    timer.Stop(printTimer)

    elapsed = timerAll.Stop(printTimer)
    distances = Util.GetAbosluteKeypointDistances(keypoints, dataset.yoloData.keypoints)
    distance = np.mean(distances)

    return rawFrame, keypoints, probs, distance, elapsed


def DetectBatchAbstract(net, rawFrame, detections: List[YoloDetection], upsampleOutput, rejectionTH,
                        upsamplingOriginalSize: bool = False, printTimer=False, upsamplingMethod: str = "bilinear"):
    testTimer = Timer()

    timerAll = Timer()
    timerAll.Start("Processing All")

    frames = []
    rawFrames = []
    paddings = []
    boundingBoxes = []
    resizeFactors = []
    upsampledHeatmaps = []

    #display = cv2.namedWindow('rawInputs', cv2.WINDOW_NORMAL)
    testTimer.Start("Yolo PostProcessing")
    for detection in detections:
        frame = detection.boundingBox.CropImage(rawFrame)

        x, padding, resizeFactor = Util.PrepareRawInput(frame, config.inputRes)
        boundingBoxes.append(detection.boundingBox)
        rawFrames.append(x)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).float()
        #x = net.PreprocessInput(x)
        frames.append(x)
        paddings.append(padding)
        resizeFactors.append(resizeFactor)

    testTimer.Stop(printTimer)

    x = torch.stack(frames)
    testTimer.Start("Prediction")
    pred = net.Predict(x)
    testTimer.Stop(printTimer)

    testTimer.Start("Restore Keypoints")

    dim = pred.shape

    '''
    display = cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
    testPred = copy.copy(pred).cpu().numpy()

    for i in range(0, dim[0]):
        heatmaps = Util.PredictionToHeatmap(testPred[i]).astype(np.float)
        heatmaps /= 255.0
        frame = rawFrames[i]
        #frame = frame.transpose(1, 2, 0)
        image = cv2.addWeighted(frame, 0.8, heatmaps, 0.2, 0)

        cv2.imshow('image2', image)
    '''

    offsetForBackground = 1

    if isinstance(net, HumanPoseNetOriginal):
        # the original model dows not include a heatmap for the background
        offsetForBackground = 0

    if not upsampleOutput:
        keypoints, probs = Util.ExtractKeypoints(pred[:, 0:dim[1] - offsetForBackground, :, :])
        keypoints = Util.RestoreKeyPoints(keypoints, resizeFactor=resizeFactors, paddings=paddings,
                                          boundingBoxes=boundingBoxes)
        keypoints = Util.FilterDetection(keypoints, probs, rejectionTH)
    else:
        fakeTargetSizes = [config.inputRes for i in range(0, dim[0])]

        if not upsamplingOriginalSize:
            upsamepledHeatmaps = Util.UpsampleBatchHeatmapsGPU(heatmaps=pred,
                                                               targetSizes=fakeTargetSizes,
                                                               method=upsamplingMethod)
        else:
            upsamepledHeatmaps = Util.UpsampleBatchHeatmapsGPU(heatmaps=pred,
                                                               targetSizes=fakeTargetSizes,
                                                               resizeFactors=resizeFactors,
                                                               method=upsamplingMethod)


        keypoints = []
        probs = []

        for i in range(0, upsamepledHeatmaps.__len__()):
            subKeypoints, subProbs = Util.ExtractKeypoints(torch.unsqueeze(upsamepledHeatmaps[i][0:dim[1] - offsetForBackground, :, :],
                                                                           dim=0))
            keypoints.append(subKeypoints[0])
            probs.append(subProbs[0])

        if not upsamplingOriginalSize:
            keypoints = Util.RestoreKeyPoints(keypoints, resizeFactor=resizeFactors,
                                              outputSize=config.inputRes,
                                              paddings=paddings,
                                              boundingBoxes=boundingBoxes)
        else:
            keypoints = Util.RestoreKeyPointsUpsampled(keypoints, paddings=paddings, boundingBoxes=boundingBoxes)

        keypoints = Util.FilterDetection(keypoints, probs, rejectionTH)

    testTimer.Stop(printTimer)

    elapsed = timerAll.Stop(printTimer)

    return keypoints, probs, elapsed, pred, paddings, upsampledHeatmaps

def PreProcessDetection(net, rawFrame, detection):
    frame = detection.boundingBox.CropImage(rawFrame)
    x, padding, resizeFactor = Util.PrepareRawInput(frame, config.inputRes)
    x = x.transpose((2, 0, 1))
    x = net.PreprocessInput(torch.from_numpy(x).float())

    return x, padding, resizeFactor

def TestMethod():
    return True

def DetectColRoPose(net, rawFrame, roposeDetections: List[YoloDetection], humanDetections: List[YoloDetection],
                    upsampleOutput, printTimer=False):
    robotPoses = []
    humanPoses = []

    #heavy parallel processing should be possible here...

    robotInputs = []
    robotFrames = []
    for detection in roposeDetections:
        inp = PreProcessDetection(net, rawFrame, detection)
        robotFrames.append(inp[0])
        robotInputs.append(inp)
    #make tensor

    humanInputs = []
    humanFrames = []
    for detection in humanDetections:
        inp = PreProcessDetection(net, rawFrame, detection)
        humanFrames.append(inp[0])
        humanInputs.append(inp)
    #make tensor

    if robotFrames.__len__() > 0:
        robotFrames = torch.stack(robotFrames)
    else:
        robotFrames = torch.tensor()

    if robotFrames.__len__() > 0:
        humanFrames = torch.stack(humanFrames)
    else:
        humanFrames = None


    prediction = net.Predict(robotFrames, humanFrames)

    test = True

    return robotPoses, humanPoses

def DetectYolo(net, x, printTimer=False):
    testTimer = Timer()
    yoloTimer = Timer()

    yoloTimer.Start("Yolo")
    detections, padding, resizeFactor = Util.PredictYolo(x, yoloNet=net, augment=False)
    testTimer.Stop()

    if detections is not None:
        humanDetections, roposeDetections = Util.FilterYolo(detections,
                                                            humanClassNr=config.yolo_HumanClassNum,
                                                            roposeClassNr=config.yolo_RoposeClassNum,
                                                            yoloInputSize=config.yolo_InputSize)
    else:
        humanDetections = roposeDetections = []

    testTimer.Start("Yolo Restore Detections")

    if humanDetections.__len__() > 0:
        for i in range(0, humanDetections.__len__()):
            humanDetections[i] = Util.RestoreYoloBB(humanDetections[i], resizeFactor, padding)

    if roposeDetections.__len__() > 0:
        for i in range(0, roposeDetections.__len__()):
            roposeDetections[i] = Util.RestoreYoloBB(roposeDetections[i], resizeFactor, padding)

    testTimer.Stop(printTimer)

    elapsed = yoloTimer.Stop(printTimer)

    return roposeDetections, humanDetections, elapsed

def DetectHumans(net, rawFrame, detections: List[YoloDetection], upsampleOutput, printTimer=False):
    return DetectBatchAbstract(net=net, rawFrame=rawFrame, detections=detections, upsampleOutput=upsampleOutput,
                          printTimer=printTimer, rejectionTH=config.humanRejectionTH)

def DetectRopose(net, rawFrame, detections: List[YoloDetection], upsampleOutput, printTimer=False, upsamplingMethod: str = "bilinear",
               upsamplingOriginalSize: bool = False):
    return DetectBatchAbstract(net=net, rawFrame=rawFrame, detections=detections, upsampleOutput=upsampleOutput,
                               rejectionTH=config.roposeRejectionTH, printTimer=printTimer,
                               upsamplingMethod=upsamplingMethod, upsamplingOriginalSize=upsamplingOriginalSize)

def TestHumanPose(net, dataset, upsampleOutput, upsamplingMethod: str = "bilinear",
                  upsamplingOriginalSize: bool = False, printTimer=False):
    return TestAbstract(net=net, dataset=dataset, upsampleOutput=upsampleOutput, upsamplingMethod=upsamplingMethod,
                        rejectionTH=config.humanRejectionTH,
                        upsamplingOriginalSize=upsamplingOriginalSize, printTimer=printTimer)

def TestRopose(net, dataset, upsampleOutput, upsamplingMethod: str = "bilinear",
               upsamplingOriginalSize: bool = False, printTimer=False):
    return TestAbstract(net=net, dataset=dataset, upsampleOutput=upsampleOutput, upsamplingMethod=upsamplingMethod,
                        rejectionTH=config.roposeRejectionTH, upsamplingOriginalSize=upsamplingOriginalSize,
                        printTimer=printTimer)