from typing import List
import torch.utils.data
from ropose_dataset_tools.DataClasses.Dataset import Dataset
from ropose.net.pytorch.TrainingHyperParams import TrainingHyperParams
from ropose.net.pytorch.Util import Util
import ropose.pytorch_config as config
from ropose.net.pytorch.DatasetTrainingUtils import DatasetUtils


class DataGenerator(torch.utils.data.dataset.Dataset):
    def __init__(self, datasets: List[type(Dataset)], trainingParametrs: TrainingHyperParams, preprocessFunction=None,
                 augmentation: bool = True):

        self.trainingParams = trainingParametrs
        self.augmentation = augmentation
        self.datasets = datasets
        self.preprocessFunction = preprocessFunction
        self.datasetUtil = DatasetUtils(onTheFlyBackgroundAugmentation=self.trainingParams.useBackgroundAugmentation,
                                        onTheFlyForegroundAugmentation=self.trainingParams.useForegroundAugmentation,
                                        useGreenscreeners=self.trainingParams.useBackgroundAugmentation or
                                                          self.trainingParams.useForegroundAugmentation)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index: int):

        if config.onTheFlyAugmentation and self.augmentation:
            # batchEntry = self.augmenter.AugmentDataset(dataset=batchEntry)
            x, x_joint, y = self.datasetUtil.LoadAugmentorDataHeatmaps(self.datasets[index])
        else:
            x, x_joint, y = self.datasetUtil.LoadXY(self.datasets[index])

        x = Util.ToFloat64Image(x)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y)

        if self.preprocessFunction is not None:
            x = self.preprocessFunction(x)

        self.datasets[index].Clear()

        if config.use3DJointInput:
            return [x, torch.from_numpy(x_joint)], y
        else:
            return x, y


