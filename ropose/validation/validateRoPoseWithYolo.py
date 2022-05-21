import sys, os
import numpy as np
from ropose.net.pytorch.RoposeNet_SimpleBaseline import RoposNet as Net
from ropose.net.pytorch.Yolo import Yolo
from ropose.net.pytorch.Util import Util, Timer
import ropose.pytorch_config as config
from ropose_dataset_tools.DataOrganizer import DataOrganizer
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from guthoms_helpers.base_types.Pose2D import Pose2D
from guthoms_helpers.base_types.Vector2D import Vector2D
from guthoms_helpers.base_types.Rotation2D import Rotation2D
from kinematic_tracker.pose_models.PoseModelRopose_2D import PoseModelRopose_2D
from ropose_dataset_tools.DataClasses import Dataset
from ropose_dataset_tools.DataSetLoader import LoadDataSet
from ropose.scripts.pytorch.TestMethods import TestRopose, DetectRopose, DetectYolo
from ropose.validation.Validator import Validator

from ropose_dataset_tools.DataClasses.DetectionTypes.YoloDetection import YoloDetection
from guthoms_helpers.filesystem.DirectoryHelper import DirectoryHelper
from guthoms_helpers.filesystem.FileHelper import FileHelper
from guthoms_helpers.common_stuff.ProgressBar import ProgressBar
from guthoms_helpers.filesystem.DirectoryHelper import DirectoryHelper

import cv2
evaluationDataBasePath = os.path.join(config.evalPath, "RoPose_Keypoints")
DirectoryHelper.CreateIfNotExist(evaluationDataBasePath)

from kinematic_tracker.tracking.Tracker import Tracker

datasets = DirectoryHelper.ListDirectories(config.roposeTestDataPath)

models = [config.roPoseNetWeights]


yolo = Yolo(netResolution=config.yolo_InputSize)
yolo.LoadPretrainedModel("/mnt/datastuff/Final_Ropose/best/Yolo.pt")

validationSet = []
resCollection = {}
overallPCKCollection = list()
punishUnknown = False
#usePose = True

trackings = [True, False]
counter = 0
for modelPath in models:
    modelName = FileHelper.GetFileName(path=modelPath, includeEnding=False)

    for tracking in trackings:
        recIuOs = []
        evaluationDataPath = os.path.join(evaluationDataBasePath, modelName, "finalApplication_" + str(tracking))

        DirectoryHelper.CreateIfNotExist(evaluationDataPath)

        keypointsPath = os.path.join(evaluationDataPath, "keypoints")
        falseNegsPath = os.path.join(evaluationDataPath, "falseNegs")
        falsePosPath = os.path.join(evaluationDataPath, "falsePos")
        probsPath = os.path.join(evaluationDataPath, "probs")
        gtPath = os.path.join(evaluationDataPath, "gt")
        fpsPath = os.path.join(evaluationDataPath, "FPS")

        if not FileHelper.FileExists(keypointsPath) or not FileHelper.FileExists(gtPath) or not \
                FileHelper.FileExists(probsPath) or not FileHelper.FileExists(fpsPath):

            if validationSet.__len__() == 0:
                for datasetPath in datasets:
                    validationSet.append(None)
                    validationSet.extend(LoadDataSet(path=datasetPath))

            net = Net(netResolution=config.inputRes)

            net.LoadPretrainedModel(modelPath)

            fps = []

            counter = 0

            validator = Validator()

            keypoints = []
            probs = []
            gts = []
            falseNegs = 0
            falsePos = 0

            meanDistances = []

            for dataset in ProgressBar(validationSet):
                counter += 1
                print("Counter: " + str(counter))
                restored = False

                roposeDetections = humanDetections = elapsed = None
                predKeypoints = predProbs = pred = paddings = upsampledHeatmaps = None

                if dataset is None:
                    roposeTracker = Tracker(similarityThreshold=0.6, iouThreshold=0.25, invalidationTimeMs=100000)
                    continue

                rawFrame = cv2.imread(dataset.rgbFrame.filePath)
                rawFrame = cv2.cvtColor(rawFrame, cv2.COLOR_BGR2RGB)

                roposeDetections, humanDetections, elapsed = DetectYolo(yolo, rawFrame,  printTimer=False)

                if tracking:
                    #recover detections if none was found
                    bbs = []
                    for roposeDetection in roposeDetections:
                        bbs.append(roposeDetection.boundingBox)

                    if counter == 135:
                        test = True

                    bbs = roposeTracker.GetNotKnownInstances(bbs, histWeightFactor=1.0, usePose=False)

                    for bb in bbs:
                        restored = True
                        roposeDetections.append(YoloDetection(bb, 1, 0.25))


                print("LENGTH: " + str(roposeDetections.__len__()))
                print("FalsePos: " + str(falsePos))
                print("FalseNeg: " + str(falseNegs))
                if roposeDetections.__len__() > 0:

                    bestDetection = roposeDetections[0]
                    falsePos += len(roposeDetections) - 1


                    predKeypoints, predProbs, _, pred, paddings, upsampledHeatmaps = DetectRopose(net, rawFrame,
                                                                                             [bestDetection],
                                                                                             upsampleOutput=True,
                                                                                             printTimer=False)
                    if tracking:
                        # create pose model
                        roposeModel = PoseModelRopose_2D()
                        roposeModel.UpdatePoses(predKeypoints[0])
                        sims = roposeTracker.AddSupervision([roposeModel])

                    if restored and not punishUnknown:

                        roposeBB = bestDetection.boundingBox
                        gtBB = dataset.yoloData.boundingBoxes[0]

                        iou = gtBB.CalculateIoU(roposeBB)
                        recIuOs.append(iou)

                        print("Rec mIou" + str(np.mean(recIuOs)))

                        distances = Util.GetAbosluteKeypointDistances(predKeypoints, dataset.yoloData.keypoints)
                        predDistance = np.mean(distances)

                        predKeypoint = []
                        for pose in predKeypoints[0]:
                            predKeypoint.append(pose.trans.toNp())

                        keypoints.append(predKeypoint)

                        probs.append(predProbs)

                        gt = []
                        for pose in dataset.rgbFrame.projectedJoints:
                            gt.append(pose.trans.toNp())

                        gts.append(gt)
                        meanDistances.append(predDistance)
                        print(meanDistances[-1])

                        fps.append(1 / elapsed)

                else:
                    falseNegs += 1
                    predKeypoints = [[]]
                    for i in range(0, 7):
                        predKeypoints[0].append(Pose2D(Vector2D(0, 0), Rotation2D(0), visible=False))

                if punishUnknown:
                    distances = Util.GetAbosluteKeypointDistances(predKeypoints, dataset.yoloData.keypoints)
                    predDistance = np.mean(distances)

                    predKeypoint = []
                    for pose in predKeypoints[0]:
                        predKeypoint.append(pose.trans.toNp())

                    keypoints.append(predKeypoint)

                    probs.append(predProbs)

                    gt = []
                    for pose in dataset.rgbFrame.projectedJoints:
                        gt.append(pose.trans.toNp())

                    gts.append(gt)
                    meanDistances.append(predDistance)
                    print(meanDistances[-1])

                    fps.append(1 / elapsed)

            keypoints = np.array(keypoints)
            gts = np.array(gts)
            probs = np.array(probs)

            FileHelper.PickleData(keypoints, keypointsPath)
            FileHelper.PickleData(falseNegs, falseNegsPath)
            FileHelper.PickleData(falsePos, falsePosPath)
            FileHelper.PickleData(gts, gtPath)
            FileHelper.PickleData(probs, probsPath)
            FileHelper.PickleData(fps, fpsPath)

        keypoints = FileHelper.DePickleData(keypointsPath)
        probs = FileHelper.DePickleData(probsPath)
        falsePos = FileHelper.DePickleData(falsePosPath)
        falseNegs = FileHelper.DePickleData(falseNegsPath)
        gts = FileHelper.DePickleData(gtPath)
        fps = FileHelper.DePickleData(fpsPath)

        mpjpe = Validator.EvaluateMPJPE(keypoints, gts)
        flatten = np.array(mpjpe).flatten()
        flatten = [a for a in flatten if (a <= 500)]
        mpjpeMean = np.mean(flatten)
        pcks = Validator.EvaluatePCKs(keypoints, gts, refIndexes=(0, 1), maxDist=1, step=0.001, setPixTH=None)

        overallPCKCollection.append({
            "pcks": pcks,
            "mpjpe": mpjpe,
            "falseNegs": falseNegs,
            "recMIoU" : np.mean(np.array(recIuOs)),
            #"pckBB": pckbb,
            #"pckStatic": pckStatic,
            "fps": fps
        })

        collection = dict()

        normCollection = dict()
        normCollection["name"] = modelName
        normCollection["falseNegs"] = float(falseNegs)
        normCollection["falsePos"] = float(falsePos)
        normCollection["meanFPS"] = float(np.mean(fps))
        normCollection["meanDist"] = float(np.mean(mpjpeMean))
        normCollection["mpjpeMean"] = float(mpjpeMean)
        normCollection["recMIoU"] = float(np.mean(np.array(recIuOs)))
        normCollection["pck20"] = pcks[0.2].tolist()
        normCollection["pck50"] = pcks[0.5].tolist()
        normCollection["pck60"] = pcks[0.6].tolist()
        normCollection["pck80"] = pcks[0.8].tolist()
        collection["norm"] = normCollection

        '''
        collectionbb = dict()
        collectionbb["name"] = modelName
        collectionbb["meanFPS"] = float(np.mean(fps))
        collectionbb["meanDist"] = float(np.mean(mpjpeMean))
        collectionbb["mpjpeMean"] = float(mpjpeMean)
        collectionbb["pck20"] = pckbb[0.2].tolist()
        collectionbb["pck50"] = pckbb[0.5].tolist()
        collectionbb["pck60"] = pckbb[0.6].tolist()
        collectionbb["pck80"] = pckbb[0.8].tolist()
        collection["bb"] = collectionbb
    
        resCollection[modelName] = collection
        '''

        print("Model: " + normCollection["name"])
        print("Mean FPS: " + str(normCollection["meanFPS"]))
        print("False Negatives: " + str(normCollection["falseNegs"]))
        print("False Pos: " + str(normCollection["falsePos"]))
        print("Mean Distances: " + str(normCollection["meanDist"]))
        print("Recovered Mean IoU: " + str(normCollection["recMIoU"]))
        #print("mpjpe: " + str(mpjpe))
        print("mpjpeMean: " + str(normCollection["mpjpeMean"]))
        print("pck20: " + str(normCollection["pck20"]))
        print("pck50: " + str(normCollection["pck50"]))
        print("pck60: " + str(normCollection["pck60"]))
        print("pck80: " + str(normCollection["pck80"]))
        #print("pckh: " + str(pckh))


        jointLabels = ["J0", "J1", "J2", "J3", "J4", "J5", "J6"]
        FileHelper.DumpDictToFile(collection, os.path.join(evaluationDataPath, "result.json"))

        dpi = 300
        #Base segment relative
        plt = Validator.PlotPCKs(pcks, index=0, all=False, title="PCKs RoPose Base", labels=["base"], showLegend=True)
        plt.savefig(os.path.join(evaluationDataPath, "pcks_base.pdf"), bbox_inches='tight', dpi=dpi, format="pdf")
        plt.savefig(os.path.join(evaluationDataPath, "pcks_base"), bbox_inches='tight', dpi=dpi)
        plt.clf()
        plt = Validator.PlotPCKs(pcks, index=6, all=False, title="PCKh RoPose EE",  labels=["End Effector"],
                                 showLegend=True)
        plt.savefig(os.path.join(evaluationDataPath, "pcks_EE.pdf"), bbox_inches='tight', dpi=dpi, format="pdf")
        plt.savefig(os.path.join(evaluationDataPath, "pcks_EE"), bbox_inches='tight', dpi=dpi)
        plt.clf()
        plt = Validator.PlotPCKs(pcks, all=True, title="PCKs RoPose All", labels=["combined"], showLegend=True)
        plt.savefig(os.path.join(evaluationDataPath, "PCKs_all_combined.pdf"), bbox_inches='tight', dpi=dpi,
                    format="pdf")
        plt.savefig(os.path.join(evaluationDataPath, "PCKs_all_combined"), bbox_inches='tight', dpi=dpi)
        plt.clf()
        plt = Validator.PlotPCKs(pcks, all=False, title="PCKs RoPose All single",
                                 labels=jointLabels, showLegend=True, inklCombined=True)
        plt.savefig(os.path.join(evaluationDataPath, "PCKs_all_single.pdf"), bbox_inches='tight', dpi=dpi,
                    format="pdf")
        plt.savefig(os.path.join(evaluationDataPath, "PCKs_all_single"), bbox_inches='tight', dpi=dpi)
        plt.clf()

        plt = Validator.PlotPCKsMean(pcks)
        plt.savefig(os.path.join(evaluationDataPath, "PCKsMean.pdf"), bbox_inches='tight', dpi=dpi,
                    format="pdf")
        plt.savefig(os.path.join(evaluationDataPath, "PCKsMean"), bbox_inches='tight', dpi=dpi)
        plt.clf()

        plt = Validator.PlotViolinEval(distances=mpjpe, maxYVal=200.0, labels=jointLabels)
        plt.savefig(os.path.join(evaluationDataPath, "violin_distances.pdf", ), bbox_inches='tight', dpi=dpi,
                    format="pdf")
        plt.savefig(os.path.join(evaluationDataPath, "violin_distances", ), bbox_inches='tight', dpi=dpi)
        plt.clf()

    # overall stuff
    # calculate both combined values
    ths = []
    withTracking = []
    withoutTracking = []
    for th in overallPCKCollection[0]["pcks"]:
        ths.append(float(th))
        withoutTracking.append(np.mean(overallPCKCollection[1]["pcks"][th]))
        withTracking.append(np.mean(overallPCKCollection[0]["pcks"][th]))

    path = os.path.join(evaluationDataBasePath, modelName, "OverallPlots", "trackingPlot")
    DirectoryHelper.CreateIfNotExist(path)

    plt = Validator.PlotPCKsSimple(ths, [withoutTracking, withTracking], labels=["No Tracking", "Tracking"])
    plt.savefig(path + ".pdf", bbox_inches='tight', dpi=dpi, format="pdf")
    plt.savefig(path + ".png", bbox_inches='tight', dpi=dpi)
    plt.clf()

    print(resCollection)



