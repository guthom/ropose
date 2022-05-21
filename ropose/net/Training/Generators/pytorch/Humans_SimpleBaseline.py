from typing import List
import torch.utils.data
import numpy as np
import os
from ropose_dataset_tools.DataClasses.Dataset import Dataset
from pycocotools.coco import COCO
from ropose.net.pytorch.TrainingHyperParams import TrainingHyperParams
import matplotlib.pyplot as plt
from ropose.net.pytorch.DatasetTrainingUtils import DatasetUtils
import ropose.pytorch_config as config
from ropose.net.pytorch.Util import Util

class DataGenerator(torch.utils.data.dataset.Dataset):

    def __init__(self, datasets: List[type(Dataset)], path: str, trainingParametrs: TrainingHyperParams,
                 preprocessFunction=None, augmentation: bool = True):

        self.trainingParams = trainingParametrs
        self.preprocessFunction = preprocessFunction
        self.augmentation = augmentation
        self.dataDir = config.cocoPath
        self.dataType = path
        self.annoFiles = []

        for type in self.dataType:
            self.annoFiles.append('{}/annotations/captions_{}.json'.format(self.dataDir, type))

        self.coco = COCO(self.annoFiles[0])

        for i in range(0, self.annoFiles.__len__()):
            tempCoco = COCO(self.annoFiles[i])
            self.coco.anns.update(tempCoco.anns)
            self.coco.catToImgs.update(tempCoco.catToImgs)
            self.coco.cats.update(tempCoco.cats)
            self.coco.imgToAnns.update(tempCoco.imgToAnns)
            self.coco.imgs.update(tempCoco.imgs)

        self.datasets = datasets

        self.catIds = config.coco_catIDs
        self.imageIDs = self.coco.getImgIds()

        self.datasetUtil = DatasetUtils(onTheFlyBackgroundAugmentation=self.trainingParams.useBackgroundAugmentation,
                                        onTheFlyForegroundAugmentation=self.trainingParams.useForegroundAugmentation,
                                        useGreenscreeners=self.trainingParams.useBackgroundAugmentation or
                                                          self.trainingParams.useForegroundAugmentation)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index: int):
        dataset = self.datasets[index]

        if dataset.annotations is not None:
            dataset.backgroundMask = self.coco.annToMask(dataset.annotations)

        if config.onTheFlyAugmentation and self.augmentation:
            # batchEntry = self.augmenter.AugmentDataset(dataset=batchEntry)
            x, x_joint, y = self.datasetUtil.LoadAugmentorDataHeatmaps(dataset)
        else:
            x, x_joint, y = self.datasetUtil.LoadXY(dataset)

        x = Util.ToFloat64Image(x)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y)

        if self.preprocessFunction is not None:
            x = self.preprocessFunction(x)

        dataset.Clear()
        return x, y


