from unittest import TestCase
import os
from ropose.net.pytorch.Util import Util
from ropose_dataset_tools.DataSetLoader import LoadDataSet
from guthoms_helpers.filesystem.FileHelper import FileHelper

from ropose_dataset_tools.DataClasses.Dataset.Dataset import Dataset
from guthoms_helpers.base_types.BoundingBox2D import BoundingBox2D as BoundingBox
from guthoms_helpers.base_types.Pose2D import Pose2D
from ropose.net.pytorch.DatasetTrainingUtils import DatasetUtils
import copy

import cv2


class GroundTruthGeneratiion(TestCase):

    def setUp(self):
        self.dataset = None
        self.LoadTestData()

    def LoadTestData(self):
        filePath = FileHelper.GetFilePath(os.path.abspath(__file__))
        filePath = os.path.join(filePath, "test_data/ropose_test_dataset/")
        self.dataset = LoadDataSet(filePath)[-1]

    def ManipulateTestDataForTestCase(self):

        datasets = []
        goldenModels = []
        dataset = copy.deepcopy(self.dataset)

        #example 1
        dataset.rgbFrame.boundingBox = BoundingBox(0, 0, 600, 600)

        dataset.rgbFrame.projectedJoints[0] = Pose2D.fromData(150, 150, 0.0)
        dataset.rgbFrame.projectedJoints[1] = Pose2D.fromData(200, 200, 0.0)
        dataset.rgbFrame.projectedJoints[2] = Pose2D.fromData(250, 250, 0.0)
        dataset.rgbFrame.projectedJoints[3] = Pose2D.fromData(300, 300, 0.0)
        dataset.rgbFrame.projectedJoints[4] = Pose2D.fromData(350, 350, 0.0)
        dataset.rgbFrame.projectedJoints[5] = Pose2D.fromData(400, 400, 0.0)
        dataset.rgbFrame.projectedJoints[6] = Pose2D.fromData(450, 450, 0.0)

        dataset.rgbFrame.resizedReprojectedPoints = self.dataset.rgbFrame.projectedJoints

        datasets.append(copy.deepcopy(dataset))

        #example 2
        dataset.rgbFrame.boundingBox = BoundingBox(0, 0, 600, 500)
        datasets.append(copy.deepcopy(dataset))

        #example 3
        dataset.rgbFrame.boundingBox = BoundingBox(0, 0, 500, 600)
        datasets.append(copy.deepcopy(dataset))

        # example 1
        goldenModel_1: Dataset = copy.deepcopy(dataset)
        goldenModel_2: Dataset = copy.deepcopy(dataset)
        goldenModel_3: Dataset = copy.deepcopy(dataset)


        goldenModel_1.rgbFrame.usedPadding = ((0, 0), (0, 0), (0, 0))
        goldenModel_1.rgbFrame.resizedReprojectedPoints[0] = Pose2D.fromData(48.0, 64.0, 0.0)
        goldenModel_1.rgbFrame.resizedReprojectedPoints[1] = Pose2D.fromData(64.0, 85.0 + 1/3, 0.0)
        goldenModel_1.rgbFrame.resizedReprojectedPoints[2] = Pose2D.fromData(80.0, 106.0 + 2/3, 0.0)
        goldenModel_1.rgbFrame.resizedReprojectedPoints[3] = Pose2D.fromData(96.0, 128.0, 0.0)
        goldenModel_1.rgbFrame.resizedReprojectedPoints[4] = Pose2D.fromData(112.0, 149.0 + 1/3, 0.0)
        goldenModel_1.rgbFrame.resizedReprojectedPoints[5] = Pose2D.fromData(128.0, 170.0 + 2/3, 0.0)
        goldenModel_1.rgbFrame.resizedReprojectedPoints[6] = Pose2D.fromData(144.0, 192.0, 0.0)
        goldenModels.append(goldenModel_1)

        #example 2
        goldenModel_2.rgbFrame.usedPadding = ((50, 50), (0, 0), (0, 0))
        goldenModel_2.rgbFrame.resizedReprojectedPoints[0] = Pose2D.fromData(48.0, 85.0 + 1/3, 0.0)
        goldenModel_2.rgbFrame.resizedReprojectedPoints[1] = Pose2D.fromData(64.0, 106.0 + 2/3, 0.0)
        goldenModel_2.rgbFrame.resizedReprojectedPoints[2] = Pose2D.fromData(80.0, 128.0, 0.0)
        goldenModel_2.rgbFrame.resizedReprojectedPoints[3] = Pose2D.fromData(96.0, 149.0 + 1/3, 0.0)
        goldenModel_2.rgbFrame.resizedReprojectedPoints[4] = Pose2D.fromData(112.0, 170.0 + 2/3, 0.0)
        goldenModel_2.rgbFrame.resizedReprojectedPoints[5] = Pose2D.fromData(128.0, 192.0, 0.0)
        goldenModel_2.rgbFrame.resizedReprojectedPoints[6] = Pose2D.fromData(144.0, 213.0 + 1/3, 0.0)
        goldenModels.append(goldenModel_2)

        # example 3
        goldenModel_3.rgbFrame.usedPadding = ((0, 0), (50, 50), (0, 0))
        goldenModel_3.rgbFrame.resizedReprojectedPoints[0] = Pose2D.fromData(64.0, 64.0, 0.0)
        goldenModel_3.rgbFrame.resizedReprojectedPoints[1] = Pose2D.fromData(80.0, 85.0 + 1/3, 0.0)
        goldenModel_3.rgbFrame.resizedReprojectedPoints[2] = Pose2D.fromData(96.0, 106.0 + 2/3, 0.0)
        goldenModel_3.rgbFrame.resizedReprojectedPoints[3] = Pose2D.fromData(112.0, 128.0, 0.0)
        goldenModel_3.rgbFrame.resizedReprojectedPoints[4] = Pose2D.fromData(128.0, 149.0 + 1/3, 0.0)
        goldenModel_3.rgbFrame.resizedReprojectedPoints[5] = Pose2D.fromData(144.0, 170.0 + 2/3, 0.0)
        goldenModel_3.rgbFrame.resizedReprojectedPoints[6] = Pose2D.fromData(160.0, 192.0, 0.0)
        goldenModels.append(goldenModel_3)

        #example 4 shifted bounding box, values etc. should stay the same
        dataset.rgbFrame.boundingBox = BoundingBox(100, 100, 700, 700)
        dataset.rgbFrame.projectedJoints[0] = Pose2D.fromData(250, 250, 0.0)
        dataset.rgbFrame.projectedJoints[1] = Pose2D.fromData(300, 300, 0.0)
        dataset.rgbFrame.projectedJoints[2] = Pose2D.fromData(350, 350, 0.0)
        dataset.rgbFrame.projectedJoints[3] = Pose2D.fromData(400, 400, 0.0)
        dataset.rgbFrame.projectedJoints[4] = Pose2D.fromData(450, 450, 0.0)
        dataset.rgbFrame.projectedJoints[5] = Pose2D.fromData(500, 500, 0.0)
        dataset.rgbFrame.projectedJoints[6] = Pose2D.fromData(550, 550, 0.0)

        datasets.append(copy.deepcopy(dataset))
        goldenModels.append(copy.deepcopy(goldenModel_1))

        dataset.rgbFrame.boundingBox = BoundingBox(100, 100, 700, 600)
        datasets.append(copy.deepcopy(dataset))
        goldenModels.append(copy.deepcopy(goldenModel_2))

        dataset.rgbFrame.boundingBox = BoundingBox(100, 100, 600, 700)
        datasets.append(copy.deepcopy(dataset))
        goldenModels.append(copy.deepcopy(goldenModel_3))

        return datasets, goldenModels

    def CompareDataset(selfd, dataset: Dataset, goldenModel: Dataset) -> bool:

        #check padding
        if dataset.rgbFrame.usedPadding != goldenModel.rgbFrame.usedPadding:
            return False

        #check poses
        for i in range(0, dataset.rgbFrame.resizedReprojectedPoints.__len__()):
            if dataset.rgbFrame.resizedReprojectedPoints[i].Rounded(decimals=3) != \
                    goldenModel.rgbFrame.resizedReprojectedPoints[i].Rounded(decimals=3):
                return False

        return True

    def test_dataLoadingWorks(self):
        self.assertIsNotNone(self.dataset)

    def test_ResizingPosesEtc(self):
        datasets, goldenModels = self.ManipulateTestDataForTestCase()

        for i in range(0, datasets.__len__()):
            dataset = datasets[i]
            goldenModel = goldenModels[i]

            img = cv2.imread(dataset.rgbFrame.filePath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = DatasetUtils.ResizeDatasetNumpy(dataset, image=img)

            # check resizing and data stuff
            self.assertTrue(img.shape == (256, 192, 3))

            self.assertTrue(self.CompareDataset(dataset, goldenModel))

    def test_NormalGTGeneration(self):

        datasetUtils = DatasetUtils(useGreenscreeners=False)
        datasets, goldenModels = self.ManipulateTestDataForTestCase()

        for dataset in datasets:
            img = cv2.imread(dataset.rgbFrame.filePath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = datasetUtils.ResizeDatasetNumpy(dataset, image=img, crop=False)
            y = datasetUtils.LoadGroundTruthToArray(dataset)

            for i in range(0, dataset.rgbFrame.resizedReprojectedPoints.__len__()):
                pose = dataset.rgbFrame.resizedReprojectedPoints[i].AsType(int)
                value = y[i][pose[1], pose[0]]
                self.assertEqual(value, 1.0)


        test = True

    def test_NormalYoloGTGeneration(self):

        datasetUtils = DatasetUtils(useGreenscreeners=False)
        datasets, goldenModels = self.ManipulateTestDataForTestCase()

        for dataset in datasets:
            img = cv2.imread(dataset.rgbFrame.filePath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = datasetUtils.ResizeDatasetNumpy(dataset, image=img, crop=True)
            y = datasetUtils.LoadGroundTruthToArray(dataset)

            for i in range(0, dataset.rgbFrame.resizedReprojectedPoints.__len__()):
                pose = dataset.rgbFrame.resizedReprojectedPoints[i].AsType(int)
                value = y[i][pose[1], pose[0]]
                self.assertEqual(value, 1.0)


        test = True


