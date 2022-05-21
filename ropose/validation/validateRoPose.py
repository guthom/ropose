import sys, os
import numpy as np
from ropose.net.pytorch.RoposeNet_SimpleBaseline import RoposNet as Net
from ropose.net.pytorch.Util import Util, Timer
import ropose.pytorch_config as config
from ropose_dataset_tools.DataOrganizer import DataOrganizer
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ropose_dataset_tools.DataClasses import Dataset
import ropose_dataset_tools.DataSetLoader as loader
from ropose.scripts.pytorch.TestMethods import TestRopose, DetectRopose
from ropose.validation.Validator import Validator

from guthoms_helpers.filesystem.DirectoryHelper import DirectoryHelper
from guthoms_helpers.filesystem.FileHelper import FileHelper
from guthoms_helpers.common_stuff.ProgressBar import ProgressBar

evaluationDataBasePath = os.path.join(config.evalPath, "RoPose_Keypoints")
DirectoryHelper.CreateIfNotExist(evaluationDataBasePath)

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]

models = [config.roPoseNetWeights]

testSettings = [
    {
        "upsampling": False,
        "upsamplingMethod": None,
        "upsamplingToOriginalSize": None
    },
    {
        "upsampling": True,
        "upsamplingMethod": "bilinear",
        "upsamplingToOriginalSize": False
    },
    {
        "upsampling": True,
        "upsamplingMethod": "bilinear",
        "upsamplingToOriginalSize": True
    },
    {
        "upsampling": True,
        "upsamplingMethod": "bicubic",
        "upsamplingToOriginalSize": False
    },
    {
        "upsampling": True,
        "upsamplingMethod": "bicubic",
        "upsamplingToOriginalSize": True
    }
]

for modelPath in models:
    validationSet = []
    resCollection = []
    overallPCKCollection = list()
    modelName = FileHelper.GetFileName(path=modelPath, includeEnding=False)

    for setting in testSettings:

        evaluationDataPath = os.path.join(evaluationDataBasePath, modelName, "_US_" +
                                          str(setting["upsampling"]) + "_METHOD_" + str(setting["upsamplingMethod"]) +
                                          "_OrigSize_" + str(setting["upsamplingToOriginalSize"]))

        DirectoryHelper.CreateIfNotExist(evaluationDataPath)

        keypointsPath = os.path.join(evaluationDataPath, "keypoints")
        probsPath = os.path.join(evaluationDataPath, "probs")
        gtPath = os.path.join(evaluationDataPath, "gt")
        fpsPath = os.path.join(evaluationDataPath, "FPS")

        if not FileHelper.FileExists(keypointsPath) or not FileHelper.FileExists(gtPath) or not \
                FileHelper.FileExists(probsPath) or not FileHelper.FileExists(fpsPath):

            if validationSet.__len__() == 0:
                colroposeDatasets = loader.LoadDataSets(config.roposeEvalDataPath, None, False)
                validationSet.extend(colroposeDatasets)

            net = Net(netResolution=config.inputRes)

            net.LoadPretrainedModel(modelPath)

            fps = []

            counter = 0

            validator = Validator()

            keypoints = []
            probs = []
            gts = []

            meanDistances = []
            print("Evaluation will be performed on " +  str(len(validationSet)) + " Validation Datasets!")
            for dataset in ProgressBar(validationSet):

                rawFrame, predKeypoints, predProbs, predDistance, elapsed = TestRopose(net, dataset,
                                                                                       upsampleOutput=setting["upsampling"],
                                                                                       upsamplingMethod=setting["upsamplingMethod"],
                                                                                       upsamplingOriginalSize=setting["upsamplingToOriginalSize"],
                                                                                       printTimer=False)


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
            FileHelper.PickleData(gts, gtPath)
            FileHelper.PickleData(probs, probsPath)
            FileHelper.PickleData(fps, fpsPath)

        keypoints = FileHelper.DePickleData(keypointsPath)
        probs = FileHelper.DePickleData(probsPath)
        gts = FileHelper.DePickleData(gtPath)
        fps = FileHelper.DePickleData(fpsPath)

        mpjpe = Validator.EvaluateMPJPE(keypoints, gts)
        flatten = np.array(mpjpe).flatten()
        flatten = [a for a in flatten if (a <= 500)]
        mpjpeMean = np.mean(flatten)
        pcks = Validator.EvaluatePCKs(keypoints, gts, refIndexes=(0, 1), maxDist=1, step=0.001, setPixTH=None)

        overallPCKCollection.append({
            "settings": setting,
            "pcks": pcks,
            "mpjpe": mpjpe,
            "fps": fps
        })

        collection = dict()

        normCollection = dict()
        normCollection["name"] = modelName
        normCollection["meanFPS"] = float(np.mean(fps))
        normCollection["meanDist"] = np.mean(mpjpe, axis=0).tolist()
        normCollection["meanDistComb"] = float(np.mean(mpjpe))
        normCollection["stdDev"] = np.std(mpjpe, axis=0).tolist()
        normCollection["stdDevComb"] = np.std(mpjpe)
        normCollection["medianDist"] = np.median(mpjpe, axis=0).tolist()
        normCollection["medianDistComb"] = float(np.median(mpjpe))
        normCollection["mpjpeMean"] = float(mpjpeMean)
        normCollection["pck20"] = pcks[0.2].tolist()
        normCollection["pck20_comb"] = np.mean(pcks[0.2]).tolist()
        normCollection["pck50"] = pcks[0.5].tolist()
        normCollection["pck50_comb"] = np.mean(pcks[0.5]).tolist()
        normCollection["pck60"] = pcks[0.6].tolist()
        normCollection["pck60_comb"] = np.mean(pcks[0.6]).tolist()
        normCollection["pck80"] = pcks[0.8].tolist()
        normCollection["pck80_comb"] = np.mean(pcks[0.8]).tolist()
        collection["norm"] = normCollection

        resCollection.append({
            "pck20_comb": normCollection["pck20_comb"],
            "pck50_comb": normCollection["pck50_comb"],
            "pck80_comb": normCollection["pck80_comb"],
            "meanDistComb": normCollection["meanDistComb"],
            "medianDistComb": normCollection["medianDistComb"],
            "stdDev": normCollection["stdDev"],
            "stdDevComb": normCollection["stdDevComb"],
        })

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
        print("Mean Distances: " + str(normCollection["meanDist"]))
        print("Mean Distances comb: " + str(normCollection["meanDistComb"]))
        print("Median Distances: " + str(normCollection["medianDist"]))
        print("Median Distances comb: " + str(normCollection["medianDistComb"]))
        #print("mpjpe: " + str(mpjpe))
        print("mpjpeMean: " + str(normCollection["mpjpeMean"]))
        print("pck20: " + str(normCollection["pck20"]))
        print("pck20_comb: " + str(normCollection["pck20_comb"]))
        print("pck50: " + str(normCollection["pck50"]))
        print("pck50_comb: " + str(normCollection["pck50_comb"]))
        print("pck60: " + str(normCollection["pck60"]))
        print("pck60_comb: " + str(normCollection["pck60_comb"]))
        print("pck80: " + str(normCollection["pck80"]))
        print("pck80_comb: " + str(normCollection["pck80_comb"]))
        #print("pckh: " + str(pckh))

        # print table
        print("Latex Table:")
        decimals = 3
        print("\\textbf & \\textbf {PCK@0.2} & \\textbf{PCK@0.5} & \\textbf{PCK@0.8} \\\\ \\midrule")
        print("\\textbf{combined} & " + str(round(normCollection["pck20_comb"], decimals)) + " & "
              + str(round(normCollection["pck50_comb"], decimals)) + " & "
              + str(round(normCollection["pck80_comb"],decimals)) + "\\\\")

        for jointNum in range(0, 7):
            print("\\textbf{J" + str(jointNum) + "} & " + str(round(normCollection["pck20"][jointNum], decimals)) + " & "
                  + str(round(normCollection["pck50"][jointNum], decimals)) + " & "
                  + str(round(normCollection["pck80"][jointNum], decimals)) + "\\\\")

        decimals=2
        print("Latex Table 2:")
        print("\\textbf {Mean}" + " & " +
              str(round(normCollection["meanDistComb"], decimals)) + " & " +
              str(round(normCollection["meanDist"][0], decimals)) + " & " +
              str(round(normCollection["meanDist"][1], decimals)) + " & " +
              str(round(normCollection["meanDist"][2], decimals)) + " & " +
              str(round(normCollection["meanDist"][3], decimals)) + " & " +
              str(round(normCollection["meanDist"][4], decimals)) + " & " +
              str(round(normCollection["meanDist"][5], decimals)) + " & " +
              str(round(normCollection["meanDist"][6], decimals)) + "\\\\")
        print("\\textbf {Median}" + " & " +
              str(round(normCollection["medianDistComb"], decimals)) + " & " +
              str(round(normCollection["medianDist"][0], decimals)) + " & " +
              str(round(normCollection["medianDist"][1], decimals)) + " & " +
              str(round(normCollection["medianDist"][2], decimals)) + " & " +
              str(round(normCollection["medianDist"][3], decimals)) + " & " +
              str(round(normCollection["medianDist"][4], decimals)) + " & " +
              str(round(normCollection["medianDist"][5], decimals)) + " & " +
              str(round(normCollection["medianDist"][6], decimals)) + "\\\\")

        print("\\textbf {Variance}" + " & " +
              str(round(np.sqrt(normCollection["stdDevComb"]), decimals))+ " & " +
              str(round(np.sqrt(normCollection["stdDev"][0]), decimals)) + " & " +
              str(round(np.sqrt(normCollection["stdDev"][1]), decimals)) + " & " +
              str(round(np.sqrt(normCollection["stdDev"][2]), decimals)) + " & " +
              str(round(np.sqrt(normCollection["stdDev"][3]), decimals)) + " & " +
              str(round(np.sqrt(normCollection["stdDev"][4]), decimals)) + " & " +
              str(round(np.sqrt(normCollection["stdDev"][5]), decimals)) + " & " +
              str(round(np.sqrt(normCollection["stdDev"][6]), decimals)) + "\\\\")

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

        plt = Validator.PlotViolinEval(distances=mpjpe, maxYVal=100.0, labels=jointLabels)
        plt.savefig(os.path.join(evaluationDataPath, "violin_distances.pdf", ), bbox_inches='tight', dpi=dpi,
                    format="pdf")
        plt.savefig(os.path.join(evaluationDataPath, "violin_distances", ), bbox_inches='tight', dpi=dpi)
        plt.clf()

        #pickle fineal overall data
        FileHelper.PickleData(overallPCKCollection,
                              os.path.join(evaluationDataBasePath, modelName, "OverallData.pickle"))

print(resCollection)


decimals = 3
print("Latex Table 3")
print("\\textbf{A} & None & $64\\times48$ " + " & " +
      str(round(resCollection[0]["pck20_comb"], decimals)) + " & " +
      str(round(resCollection[0]["pck50_comb"], decimals)) + " & " +
      str(round(resCollection[0]["pck80_comb"], decimals)) + "\\\\")

print("\\textbf{B} & Bilinear & $256\\times192$ " + " & " +
      str(round(resCollection[1]["pck20_comb"], decimals)) + " & " +
      str(round(resCollection[1]["pck50_comb"], decimals)) + " & " +
      str(round(resCollection[1]["pck80_comb"], decimals)) + "\\\\")

print("\\textbf{C} & Bilinear & Original Size " + " & " +
      str(round(resCollection[2]["pck20_comb"], decimals)) + " & " +
      str(round(resCollection[2]["pck50_comb"], decimals)) + " & " +
      str(round(resCollection[2]["pck80_comb"], decimals)) + "\\\\")

print("\\textbf{D} & Bicubic & $256\\times192$ " + " & " +
      str(round(resCollection[3]["pck20_comb"], decimals)) + " & " +
      str(round(resCollection[3]["pck50_comb"], decimals)) + " & " +
      str(round(resCollection[3]["pck80_comb"], decimals)) + "\\\\")

print("\\textbf{E} & Bicubic & Original Size " + " & " +
      str(round(resCollection[4]["pck20_comb"], decimals)) + " & " +
      str(round(resCollection[4]["pck50_comb"], decimals)) + " & " +
      str(round(resCollection[4]["pck80_comb"], decimals)) + "\\\\")

print("Latex Table 4:")
print("\\textbf{mean} &" +
      str(round(resCollection[0]["meanDistComb"], decimals)) + " & " +
      str(round(resCollection[1]["meanDistComb"], decimals)) + " & " +
      str(round(resCollection[2]["meanDistComb"], decimals)) + " & " +
      str(round(resCollection[3]["meanDistComb"], decimals)) + " & " +
      "\\textbf{" + str(round(resCollection[4]["meanDistComb"], decimals)) + "}\\\\")

print("\\textbf{median} &" +
      str(round(resCollection[0]["medianDistComb"], decimals)) + " & " +
      str(round(resCollection[1]["medianDistComb"], decimals)) + " & " +
      str(round(resCollection[2]["medianDistComb"], decimals)) + " & " +
      str(round(resCollection[3]["medianDistComb"], decimals)) + " & " +
      "\\textbf{" + str(round(resCollection[4]["medianDistComb"], decimals)) + "}\\\\")
