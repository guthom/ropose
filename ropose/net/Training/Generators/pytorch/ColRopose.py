from typing import List
import torch.utils.data
import os
from ropose_dataset_tools.DataClasses.Dataset import Dataset

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from ropose.net.pytorch.Util import Util
import ropose.pytorch_config as config


class DataGenerator(torch.utils.data.dataset.Dataset):
    def __init__(self, roPoseDatasets: List[type(Dataset)], humanDatasets: List[type(Dataset)], path: str,
                 preprocessFunction=None):
        self.preprocessFunction = preprocessFunction
        self.dataDir = config.cocoPath
        self.dataType = path
        self.annoFile = '{}/annotations/instances_{}.json'.format(self.dataDir, self.dataType)
        self.coco = COCO(self.annoFile)

        self.catIds = self.coco.getCatIds(catNms=['person'])
        self.imageIDs = self.coco.getImgIds(catIds=self.catIds)

        self.roPoseDatasets = roPoseDatasets
        self.humanDatasets = humanDatasets

    def ShuffleHumanSets(self):
        self.humanDatasets = Util.SwapDatasetData(self.humanDatasets)

    def __len__(self):
        return len(self.roPoseDatasets)

    def __getitem__(self, index: int):

        if self.humanDatasets[index].annotations is not None:
            self.humanDatasets[index].backgroundMask = self.coco.annToMask(self.humanDatasets[index].annotations)

        if config.onTheFlyAugmentation:
            # batchEntry = self.augmenter.AugmentDataset(dataset=batchEntry)
            x, x_joint, y = Util.LoadAugmentedData(self.humanDatasets[index])
        else:
            x, x_joint, y = Util.LoadXY(self.humanDatasets[index])

        y_2 = torch.from_numpy(y)
        x = x.transpose((2, 0, 1))
        x_2 = torch.from_numpy(x).float()

        if config.onTheFlyAugmentation:
            # batchEntry = self.augmenter.AugmentDataset(dataset=batchEntry)
            x, x_joint, y = Util.LoadAugmentedData(self.roPoseDatasets[index])
        else:
            x, x_joint, y = Util.LoadXY(self.roPoseDatasets[index])

        y_1 = torch.from_numpy(y)
        x = x.transpose((2, 0, 1))
        x_1 = torch.from_numpy(x).float()

        if self.preprocessFunction is not None:
            x_1 = self.preprocessFunction(x_1)
            x_2 = self.preprocessFunction(x_2)

        self.humanDatasets[index].Clear()
        self.roPoseDatasets[index].Clear()


        return x_1, y_1, x_2, y_2


