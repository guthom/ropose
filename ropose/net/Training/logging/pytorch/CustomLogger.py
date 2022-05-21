from ropose_dataset_tools.DataClasses.Dataset import Dataset
import os, sys
import time
from typing import List
from ropose.net.pytorch.Util import Util
from termcolor import colored

import json

class CustomLogger(object):

    def __init__(self, snapshootDir: str, resultDir: str, modelPath: str, model: 'NetBase', datasets: Dataset, period=1):
        """Create a summary writer logging to log_dir."""
        self.snapshootDir = snapshootDir
        self.resultDir = resultDir
        self.logPath = os.path.join(resultDir, "trainingLog.json")
        self.currentModelPath = os.path.join(resultDir, "CurrentModel.pt")

        self.logList = []

        self.modelPath = modelPath
        self.model = model
        self.period = period
        self.datasets = datasets
        self.bestLoss = sys.float_info.max

    def CheckDirs(self):
        if not os.path.isdir(self.resultDir):
            os.makedirs(self.resultDir, exist_ok=True)
        if not os.path.isdir(self.snapshootDir):
            os.makedirs(self.snapshootDir, exist_ok=True)

    def SaveSnapshoot(self, epoch: int):
        if os.path.isfile(self.modelPath):
            newFilePath = os.path.join(self.snapshootDir, "snapshot_" + str(epoch-1) + ".pt")
            os.rename(self.modelPath, newFilePath)
            time.sleep(1)

    def SaveLatestModel(self):
        self.model.SaveModelWeights(self.currentModelPath)

    def SaveNewBest(self, epoch: int, loss: float):
        self.bestLoss = loss
        self.model.SaveModelWeights()

    def SaveResult(self, epoch: int, latestLoss=float, partLosses: List[float] = None, learningRate=None):

        entry = dict()
        entry["loss"] = latestLoss
        entry["partLoss"] = partLosses
        entry["epoch"] = epoch
        entry["learningRate"] = learningRate

        self.logList.append(entry)

        with open(self.logPath, 'w') as outfile:
            json.dump(self.logList, outfile)
        print("Update log at " +  self.logPath)

        '''
        singleSets = []
        for dataset in self.datasets:
            singleSet, buf = Util.PlotRoPoseSet(dataset=dataset, neuralNet=self.model, upsampling=True,
                                                preprocessFunction=self.model.PreprocessInput)

            singleSets.append(singleSet)

        for singleSet in singleSets:
            files = os.listdir(self.resultDir)

            if index is not None:
                addInfos = "_" + str(index)
            else:
                addInfos = ""

            fileName = "result" + addInfos + str(files.__len__()) + ".png"
            singleSet.save(os.path.join(self.resultDir, fileName))
        '''

    def EpochEnd(self, epoch: int, latestLoss: float, partLosses = None, learningRate = None):
        self.SaveLatestModel()
        if latestLoss < self.bestLoss:
            self.SaveNewBest(epoch, latestLoss)
            print("New Model Saved, new best Loss: " + str(self.bestLoss))

        self.SaveResult(epoch, latestLoss, partLosses, learningRate)
