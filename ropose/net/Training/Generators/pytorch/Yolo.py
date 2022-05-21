from typing import List
import torch.utils.data
from ropose_dataset_tools.DataClasses.Dataset import Dataset
from ropose.net.pytorch.TrainingHyperParams import TrainingHyperParams
from ropose.net.pytorch.Util import Util
import ropose.pytorch_config as config

from ropose.net.pytorch.DatasetTrainingUtils import DatasetUtils

class DataGenerator(torch.utils.data.dataset.Dataset):
    def __init__(self, datasets: List[type(Dataset)], trainingParametrs: TrainingHyperParams, preprocessFunction=None,
                 augment: bool = True):

        self.augment = augment
        self.trainingParams = trainingParametrs
        self.datasets = datasets
        self.preprocessFunction = preprocessFunction
        self.datasetUtil = DatasetUtils(onTheFlyBackgroundAugmentation=self.trainingParams.useBackgroundAugmentation,
                                        onTheFlyForegroundAugmentation=self.trainingParams.useForegroundAugmentation,
                                        useGreenscreeners=self.trainingParams.useBackgroundAugmentation or
                                                          self.trainingParams.useForegroundAugmentation)

    def __len__(self):
        return len(self.datasets)

    @staticmethod
    def collate_fn(batch):
        #see https://github.com/ultralytics/yolov3/blob/e42278e9812f1790e5d8ea1bc83eab3de95a670e/utils/datasets.py#L298
        img, label = list(zip(*batch))# transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0)


    def __getitem__(self, index: int):

        if self.augment:
            # batchEntry = self.augmenter.AugmentDataset(dataset=batchEntry)
            x, y = self.datasetUtil.LoadAugmentorYolo(self.datasets[index],
                                                      foregroundAugmentation=self.trainingParams.useForegroundAugmentation)
        else:
            x, y = self.datasetUtil.LoadYoloData(self.datasets[index])

        x = Util.ToFloat64Image(x)

        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y)

        if self.preprocessFunction is not None:
            x = self.preprocessFunction(x)

        return x, y


