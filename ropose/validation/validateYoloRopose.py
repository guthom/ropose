import ropose.scripts.BootstrapDL
import os
import numpy as np
from ropose.net.pytorch.Util import Util, Timer

import ropose.pytorch_config as config
import matplotlib.pyplot as plt
import ropose_dataset_tools.DataSetLoader as loader
import ropose_dataset_tools.CocoSetLoader as cocoLoader
from ropose_dataset_tools.DataOrganizer import DataOrganizer
from ropose.validation.Validator import Validator
from ropose.validation.validateYoloCommons import ValidateYolo

from guthoms_helpers.filesystem.DirectoryHelper import DirectoryHelper
from guthoms_helpers.filesystem.FileHelper import FileHelper

evaluationDataBasePath = os.path.join(config.evalPath, "RoPose_Yolo")
DirectoryHelper.CreateIfNotExist(evaluationDataBasePath)

#models = DirectoryHelper.ListDirectoryFiles("/mnt/datastuff/TrainedModels/Yolo_new", fileEndings=[".pt"])

models = [config.roPoseYoloWeights]

validationSet = []
resCollection = {}

counter = -1

for modelPath in models:
    counter += 1

    modelName = FileHelper.GetFileName(path=modelPath, includeEnding=False)
    evaluationDataPath = os.path.join(evaluationDataBasePath, modelName)
    DirectoryHelper.CreateIfNotExist(evaluationDataPath)

    detecionsPath = os.path.join(evaluationDataPath, "detections")
    gtPath = os.path.join(evaluationDataPath, "gt")
    fpsPath = os.path.join(evaluationDataPath, "FPS")

    if not FileHelper.FileExists(detecionsPath) or not FileHelper.FileExists(gtPath) or not \
        FileHelper.FileExists(fpsPath):

        if validationSet.__len__() == 0:
            datasets = []
            roPoseDatasets = loader.LoadDir(path=config.roposeEvalDataPath)
            for set in roPoseDatasets:
                validationSet.extend(set)

        fps, yoloDetections, yoloGTs = ValidateYolo(validationSet, modelPath=modelPath)

        FileHelper.PickleData(yoloDetections, detecionsPath)
        FileHelper.PickleData(yoloGTs, gtPath)
        FileHelper.PickleData(fps, fpsPath)

    yoloDetections = FileHelper.DePickleData(detecionsPath)
    yoloGTs = FileHelper.DePickleData(gtPath)
    fps = FileHelper.DePickleData(fpsPath)

    ious = Validator.EvaluateIoU(yoloDetections, yoloGTs)
    miou = np.mean(ious[config.yolo_RoposeClassNum])
    medianIou = np.median(ious[config.yolo_RoposeClassNum])

    aps = dict()
    results = dict()

    #for th in np.arange(0.0, 1.025, 0.025):
    for th in np.arange(0.0, 1.025, 0.025):
        results[th] = Validator.NewEvaluateBinary(yoloDetections, yoloGTs, threshold=th)

    aps, f1Scores, recalls, precisions, areas = Validator.Evaluate_PrecissionRecall_Binary(results, path=evaluationDataPath,
                                                     classIndexes=[config.yolo_RoposeClassNum])

    for th in np.arange(0.0, 1.025, 0.025):
      print(str(th) + ": " + str(aps[config.yolo_RoposeClassNum][th]))

    for j in range(0, len(f1Scores)):
      print(str(j) + ": " + str(f1Scores[j]))




    #corrects, falsePos, falseNeg = Validator.EvaluateFalseNegPos(yoloDetections, yoloGTs)
    #aps = Validator.EvaluateAP(yoloDetections, yoloGTs, 0.01)

    #Validator.Plot_mAP(aps, evaluationDataPath, classIndexes=[2])


    print("Mean FPS: " + str(np.mean(fps)))
    print("mIoU: " + str(miou))
    print("medianIoU: " + str(medianIou))
    #print("Corrects: " + str(corrects[81]))
    #print("FalseNeg: " + str(falseNeg[81]))
    #print("FalsePos: " + str(falsePos[81]))
    print("mAP@0.5: " + str(aps[config.yolo_RoposeClassNum][0.5]))

    plt = Validator.PlotmAPs(aps, classIndexes=[config.yolo_RoposeClassNum])
    plt.savefig(os.path.join(evaluationDataPath, "mAPs"), bbox_inches='tight', dpi=300)
    plt.clf()
    decimals = 3
    print("Latextable:")
    print("\\textbf{Results} & "
          + str(round(areas[config.yolo_RoposeClassNum], decimals)) + " & "
          + str(round(f1Scores[8], decimals)) + " & "
          + str(round(f1Scores[20], decimals)) + " & "
          + str(round(f1Scores[32], decimals)) + " & "
          + str(round(float(miou), decimals)) + "\\\\")
