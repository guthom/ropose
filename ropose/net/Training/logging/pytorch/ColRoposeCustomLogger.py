from ropose_dataset_tools.DataClasses.Dataset import Dataset
import os
import time
import ropose.pytorch_config as config
from pycocotools.coco import COCO
from ropose.net.pytorch.Util import Util

class CustomLogger(object):

    def __init__(self, snapshootDir: str, resultDir: str, modelPath: str, model: 'NetBase', datasets: Dataset, period=1):
        """Create a summary writer logging to log_dir."""
        self.snapshootDir = snapshootDir
        self.resultDir = resultDir
        self.modelPath = modelPath
        self.model = model
        self.period = period
        self.annoFile = '{}/annotations/instances_{}.json'.format(config.cocoPath, "train2017")
        self.coco = COCO(self.annoFile)
        self.datasets = []
        for dataset in datasets:
            if dataset.annotations is not None:
                dataset.backgroundMask = self.coco.annToMask(dataset.annotations)
            self.datasets.append(dataset)

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

    def SaveResult(self, index=None):
        singleSets = []

        for dataset in self.datasets:
            imgs, bufs = Util.PlotColPoseSet(dataset=dataset, neuralNet=self.model, upsampling=True,
                                           preprocessFunction=self.model.PreprocessInput)
            for img in imgs:
                singleSets.append(img)

        for singleSet in singleSets:
            files = os.listdir(self.resultDir)

            if index is not None:
                addInfos = "_" + str(index)
            else:
                addInfos = ""

            fileName = "result" + addInfos + str(files.__len__()) + ".png"
            singleSet.save(os.path.join(self.resultDir, fileName))

    def EpochEnd(self, epoch: int):
        if epoch % self.period == 0:
            self.SaveSnapshoot(epoch)
        self.model.SaveModelWeights()
        if self.datasets is not None:
            self.SaveResult()
