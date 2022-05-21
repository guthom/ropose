import sys, os
import numpy as np
from ropose_dataset_tools.DataClasses.Dataset import Dataset
from guthoms_helpers.base_types.Pose2D import Pose2D
from guthoms_helpers.base_types.Vector2D import Vector2D
from guthoms_helpers.base_types.BoundingBox2D import BoundingBox2D as BoundingBox
from guthoms_helpers.common_stuff.Timer import Timer
from ropose_greenscreener.ImageGreenscreener import ImageGreenscreener
from ropose_greenscreener.RoposeGreenscreener import RoposeGreenscreener
from ropose.net.pytorch.Util import Util as util
import ropose.pytorch_config as config
import cv2
import random
import matplotlib.pyplot as plt
from collections import Iterable
from typing import Optional
import numpy as np
from skimage import measure, io
import copy
import time
import scipy.ndimage.filters as fi
from ropose_dataset_tools.Augmentation.Augmenter import Augmenter
from ropose_dataset_tools.Augmentation.Augmentor import Augmentor
from skimage.transform import resize

class DatasetUtils(object):

    augmentor = Augmentor()

    def __init__(self, onTheFlyBackgroundAugmentation:bool =config.onTheFlyBackgroundAugmentation,
                 onTheFlyForegroundAugmentation: bool = config.onTheFlyForegroundAugmentation,
                 useGreenscreeners: Optional[bool] = None):

        if useGreenscreeners is None:
            useGreenscreeners = onTheFlyBackgroundAugmentation or onTheFlyForegroundAugmentation

        self.onTheFlyForegroundAugmentation = onTheFlyForegroundAugmentation
        self.onTheFlyForegroundAugmentation = onTheFlyBackgroundAugmentation

        if useGreenscreeners:
            if onTheFlyBackgroundAugmentation:
                self.greenscreener = ImageGreenscreener(usePreloader=False)

            if onTheFlyForegroundAugmentation:
                self.roposeGreenscreener = RoposeGreenscreener(os.path.join(config.simDataPath, "ropose_simdata_10K/"),
                                                               usePreloader=False)

        self.timer = Timer()


    def LoadImagesToArray(self, datasets, exchangeForeground: bool = False):
        imageArray = []

        if isinstance(datasets, Iterable):
            for dataset in datasets:
                imageArray.append(self.LoadImageToArray(dataset, exchangeForeground=exchangeForeground))
        else:
            imageArray.append(self.LoadImageToArray(datasets, exchangeForeground=exchangeForeground))

        return imageArray

    def LoadImageToArray(self, dataset: Dataset, exchangeForeground: bool = False):

        img = cv2.imread(dataset.rgbFrame.filePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if exchangeForeground and Augmenter.DecideByProb(config.onTheFlyForegroundAugmentationProb):
            count = random.randint(1, config.forgroundAugmentationMaxCount)
            for i in range(0, count):
                img, yoloData = self.roposeGreenscreener.AddForeground(img)
                dataset.yoloData.Extend(yoloData)

        if config.onTheFlyBackgroundAugmentation and dataset.metadata.greenscreened:
            img = self.greenscreener.AddBackground(img)

        if config.loadGrayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = self.ResizeDatasetNumpy(dataset, image=img)

        if config.useHSVColorspace:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        return img

    def LoadYoloImagesToArray(self, dataset: Dataset, exchangeForeground: bool = False):
        img = cv2.imread(dataset.rgbFrame.filePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if exchangeForeground:
            count = random.randint(1, config.forgroundAugmentationMaxCount)
            for i in range(0, count):
                img, yoloData = self.roposeGreenscreener.AddForeground(img)

                dataset.yoloData.Extend(yoloData)

        if config.onTheFlyBackgroundAugmentation and dataset.metadata.greenscreened:
            img = self.greenscreener.AddBackground(img)

        if config.loadGrayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if config.useHSVColorspace:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        img = self.ResizeDatasetNumpy(dataset, image=img, size=config.yolo_InputSize, crop=False)

        return img

    def LoadRawImageToArray(self, dataset, crop=True, exchangeForeground: bool = False):

        img = cv2.imread(dataset.rgbFrame.filePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if exchangeForeground:
            count = random.randint(1, config.forgroundAugmentationMaxCount)
            for i in range(0, count):
                img, yoloData = self.roposeGreenscreener.AddForeground(img)
                dataset.yoloData.Extend(yoloData)

        if config.onTheFlyBackgroundAugmentation and dataset.metadata.greenscreened:
            img = self.greenscreener.AddBackground(img)

        if config.useHSVColorspace:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        if config.cropInputToBB and crop:
            img = util.CropImage(image=img, bb=dataset.rgbFrame.boundingBox)

        return img

    def LoadYoloData(self, dataset: Dataset):

        myDataset = copy.deepcopy(dataset)

        x = self.LoadYoloImagesToArray(myDataset)
        y = self.LoadYoloBoundingBox(myDataset, x)

        return np.array(x), np.array(y)


    def LoadAugmentedData(self, dataset: Dataset, inputRes=config.inputRes, outputRes=config.outputRes,
                          exchangeForeground: bool = False, useRandomErasing: bool=config.useRandomErasing):

        myDataset = copy.deepcopy(dataset)

        #myDataset.backgroundMask = myDataset.GetBackgroundMaskURDF()

        augCollection = Augmenter.GetRandomValues(config.inputRes, config.outputRes)
        M_img, _ = Augmenter.Affine(augCollection, inputRes, outputRes)

        x = self.LoadImagesToArray(myDataset, exchangeForeground)[0]

        if useRandomErasing and augCollection["randomErasing"]:
            x = Augmenter.AddRandomErasing(x, probabilty=config.randomErasingProb,
                                           maxObjectCount=config.randomErasingMaxObjectCount,
                                           coverRange=config.randomErasingMaxAreaCover)


        x = Augmenter.AugmentImg(x, M_img)

        y = DatasetUtils.CreateHeatmapFromGroundTruth(myDataset, augMat=M_img)

        if config.use3DJointInput:
            x_joint = DatasetUtils.Load3DJointPos(myDataset)
        else:
            x_joint = None

        ''' #Debug View
        allHeatmaps = np.sum(y, axis=0)
        allHeatmaps = util.HeatmapToCV(allHeatmaps)
        img = util.NPToCVImage(x)
        combi = util.CreateOverlayingImage(img, allHeatmaps, 0.25)

        plt.imshow(combi)
        plt.show()
        '''
        #del myDataset

        return np.array(x), x_joint, y

    def LoadAugmentorDataBB(self, dataset: Dataset):

        myDataset = copy.deepcopy(dataset)

        x = self.LoadImagesToArray(myDataset, False)[0]
        y = DatasetUtils.CreateHeatmapFromGroundTruth(myDataset)

        if config.use3DJointInput:
            x_joint = DatasetUtils.Load3DJointPos(myDataset)
        else:
            x_joint = None

        x, y, = self.augmentor.AugmentImagesAndBBs(x, y)

        return x, x_joint, y

    def LoadAugmentorDataHeatmaps(self, dataset: Dataset):

        myDataset = copy.deepcopy(dataset)

        x = self.LoadImagesToArray(myDataset, False)[0]
        y = DatasetUtils.CreateHeatmapFromGroundTruth(myDataset)

        if config.use3DJointInput:
            x_joint = DatasetUtils.Load3DJointPos(myDataset)
        else:
            x_joint = None
        x, y, = self.augmentor.AugmentImagesAndHeatmaps(x, y)

        '''
        # allHeatmaps = np.sum(y, axis=0)
        allHeatmaps = np.sum(y[:y.shape[0], :, :], axis=0)
        allHeatmaps = util.HeatmapToCV(allHeatmaps)
        img = util.NPToCVImage(x)
        combi = util.CreateOverlayingImage(img, allHeatmaps, 0.25)
        # draw poses
        rawImage = myDataset.rgbFrame
        # combi = util.DrawCircles(combi, myDataset.rgbFrame.resizedReprojectedPoints)
        plt.imshow(combi)
        plt.savefig("/mnt/datastuff/TestExamples/test.jpg")
        '''

        return x, x_joint, y

    def LoadXY(self, dataset, load3DJoints: bool = config.use3DJointInput):

        myDataset = copy.deepcopy(dataset)

        x = self.LoadImagesToArray(myDataset)[0]

        if load3DJoints:
            x_joint = self.Load3DJointPos(myDataset)
        else:
            x_joint = None

        y = self.LoadGroundTruthToArray(myDataset)

        '''
        #allHeatmaps = np.sum(y, axis=0)
        allHeatmaps = np.sum(y[:y.shape[0], :, :], axis=0)
        allHeatmaps = util.HeatmapToCV(allHeatmaps)
        img = util.NPToCVImage(x)
        combi = util.CreateOverlayingImage(img, allHeatmaps, 0.25)
        #draw poses
        rawImage = myDataset.rgbFrame
        #combi = util.DrawCircles(combi, myDataset.rgbFrame.resizedReprojectedPoints)
        plt.imshow(combi)
        plt.savefig("/mnt/datastuff/TestExamples/human/human_test.jpg")
        '''

        return x, x_joint, y


    @staticmethod
    def LoadYoloBoundingBox(dataset: Dataset, x: np.array):
        h, w, _ = x.shape

        filled_labels = []

        for i in range(0, dataset.yoloData.resizedBoundingBoxes.__len__()):
            label = []
            bb = copy.deepcopy(dataset.yoloData.resizedBoundingBoxes[i])
            bb = bb.NormWithImage(x)

            if bb.midX > 1.0 or bb.midY > 1.0:
                label = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                filled_labels.append(np.array(label))
                continue

            bb = bb.Clip(1.0)
            classID = dataset.yoloData.classIDs[i]

            '''
            labels[:, 1] = ((x1 + x2) / 2) / padded_w -> 
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
            '''

            label.append(0.0)
            label.append(float(classID))
            label.append(bb.midX)
            label.append(bb.midY)
            label.append(bb.width)
            label.append(bb.height)

            # Hack for the used YOLO implementation, i don't know why it fails when the label is as big as the images
            # witdth or heigth

            if label[2] >= 1.0:
                label[2] *= 0.9999

            if label[3] >= 1.0:
                label[3] *= 0.9999

            #ignore faulty lables
            if label[1] < 0.0 or label[2] <= 0.0 or label[3] <= 0.0 or label[4] <= 0.0 or label[5] <= 0.0:
                label = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            filled_labels.append(np.array(label))
        return np.array(filled_labels)

    def LoadAugmentedYolo(self, dataset: Dataset, foregroundAugmentation:bool=False,
                          useRandomErasing:bool=config.useRandomErasing):

        myDataset = copy.deepcopy(dataset)

        augCollection = Augmenter.GetRandomValues(config.yolo_InputSize)
        x = self.LoadYoloImagesToArray(myDataset, foregroundAugmentation)
        M_img, M_gt = Augmenter.Affine(augCollection)
        x = Augmenter.AugmentImg(x, M_img)
        myDataset = Augmenter.AugmentYoloData(myDataset, M_img)

        if useRandomErasing and augCollection["randomErasing"]:
            x = Augmenter.AddRandomErasing(x, probabilty=config.randomErasingProb,
                                           maxObjectCount=config.randomErasingMaxObjectCount,
                                           coverRange=config.randomErasingMaxAreaCover)

        y = self.LoadYoloBoundingBox(myDataset, x)


        '''

        fig, ax = plt.subplots(1)
        #rawImage = util.LoadRawImageToArray(myDataset, False)
        #rawImage = Augmenter.AugmentImg(rawImage, M_img)
        plt.imshow(x)
        filename = "/mnt/datastuff/TestStuff/yoloAugmentedGT.jpg"
        for i in range(0, myDataset.yoloData.resizedBoundingBoxes.__len__()):
            boudningBox = myDataset.yoloData.resizedBoundingBoxes[i]
            boudningBox.AddPatch(plt, ax, config.yolo_Classes[myDataset.yoloData.classIDs[i]]["name"])
            plt.savefig(filename)
        time.sleep(0.25)
        '''

        return np.array(x), np.array(y)

    def LoadAugmentorYolo(self, dataset: Dataset, foregroundAugmentation:bool=False):

        myDataset = copy.deepcopy(dataset)

        x = self.LoadYoloImagesToArray(myDataset, foregroundAugmentation)
        #x_raw = cv2.imread(myDataset.rgbFrame.filePath)

        tempBB = []

        for bb in myDataset.yoloData.resizedBoundingBoxes:
            tempBB.append(bb.ToIaaBoundingBox())

        x, y = self.augmentor.AugmentImagesAndBBs(np.array([x]), tempBB)

        for i in range(0, len(y)):
            myDataset.yoloData.resizedBoundingBoxes[i] = BoundingBox.FromIaaBB(y[i])

        y = self.LoadYoloBoundingBox(myDataset, x[0])

        return np.array(x[0]), np.array(y)

    def LoadAugmentedYoloDemo(self, dataset: Dataset, foregroundAugmentation:bool=False):

        myDataset = copy.deepcopy(dataset)

        augCollection = Augmenter.GetRandomValues(config.yolo_InputSize)
        x = self.LoadYoloImagesToArray(myDataset, foregroundAugmentation)
        M_img, M_gt = Augmenter.Affine(augCollection)
        x = Augmenter.AugmentImg(x, M_img)
        myDataset = Augmenter.AugmentYoloData(myDataset, M_img)
        y = self.LoadYoloBoundingBox(myDataset, x)

        return np.array(x), np.array(y), myDataset

    @staticmethod
    def LoadGroundTruthToArray(datasets):

        heatmaps = []
        if isinstance(datasets, Iterable):
            for dataset in datasets:
                heatmap = DatasetUtils.CreateHeatmapFromGroundTruth(dataset)
                heatmaps.append(heatmap)
        else:
            heatmaps = DatasetUtils.CreateHeatmapFromGroundTruth(datasets)

        return np.array(heatmaps)


    @staticmethod
    def CreateHeatmapFromGroundTruth(dataset: Dataset, augMat=None):
        timer = Timer()
        resolution = config.inputRes

        #timer.Start("Upsampling")
        if not config.includeUpsampling:
            outputRes = config.outputRes
        else:
            outputRes = resolution

        factor = int(resolution[0] / outputRes[0])
        poses = dataset.rgbFrame.resizedReprojectedPoints


        #timer.Stop(True)

        heatmaps = []



        for i in range(0, poses.__len__()):
            heatmap = np.zeros(tuple(config.inputRes))

            for j in range(0, poses.__len__()):
                    pose = poses[j].AsType(int)
                    x = pose[1]
                    y = pose[0]

                    if(x in range(0, config.inputRes[0]) and y in range(0, config.inputRes[1])):
                        #set to max value if pose is our joint of intresst
                        #set to min if other joint
                        if x and y != np.nan:
                            if i == j:
                                heatmap[x][y] = config.heatmapMaxValue

            if heatmap.shape[0] > outputRes[0]:
                #timer.Start("Block Reduce")
                heatmap = measure.block_reduce(heatmap, (factor, factor), np.mean,
                                               cval=config.heatmapMinValue)
                #timer.Stop(True)

            #timer.Start("Gaussian Filter")
            heatmap = fi.gaussian_filter(heatmap, config.gaussSigma)
            #timer.Stop(True)

            heatmap = util.normalizeHeatmap(heatmap)
            heatmaps.append(heatmap)

        if dataset.backgroundMask is not None:
            #background = np.ones(dataset.backgroundHeatmap.shape) - dataset.backgroundHeatmap
            background = np.ones(dataset.backgroundHeatmap.shape, dtype=float)
            background = np.array(background).astype(float)
            background -= dataset.backgroundHeatmap

            background = measure.block_reduce(background, (factor, factor), np.mean, cval=config.heatmapMinValue)
            background = util.normalizeHeatmap(background)

            if augMat is not None:
                background = Augmenter.AugmentHeatmap(background, augMat, config.heatmapMinValue)

        else:
            #create backgrund heatmap could be combined with the code above
            background = np.ones(tuple(config.inputRes))
            #TODO: Develop better algorythim
            for i in range(0, poses.__len__()-1):
                pose1 = poses[i].AsType(int)
                pose2 = poses[i+1].AsType(int)

                x1 = pose1[1]
                y1 = pose1[0]
                x2 = pose2[1]
                y2 = pose2[0]

                if x1 and x2 and y1 and y2 >= 0.0:
                    xrange = abs(x1 - x2)
                    yrange = abs(y1 - y2)

                    if(xrange > 0.0):
                        m = (y1 - y2)/(x1 - x2)
                    else:
                        m = 0

                    b = y1 - m*x1

                    step = 1
                    if xrange >= yrange:
                        if x1 > x2:
                            step = -1
                        for x in range(x1, x2, step):
                            y = int(m*x + b)

                            if x in range(0, config.inputRes[0]) and y in range(0, config.inputRes[1]):
                                if x and y != np.nan:
                                    background[x][y] = config.heatmapMinValue
                    else:
                        if y1 > y2:
                            step = -1
                        for y in range(y1, y2, step):
                            if m != 0.0:
                                x = (y - b) / m
                                x = int(x)
                            else:
                                x = y

                            if x in range(0, config.inputRes[0]) and y in range(0, config.inputRes[1]):
                                if x and y != np.nan:
                                    background[x][y] = config.heatmapMinValue

            if augMat is not None:
                background = Augmenter.AugmentHeatmap(background, augMat, config.heatmapMaxValue)

            background = fi.gaussian_filter(background, config.gaussSigma * 3)

            if background.shape[0] > outputRes[0]:
                factor = int(resolution[0] / outputRes[0])
                background = measure.block_reduce(background, (factor, factor), np.mean, cval=config.heatmapMaxValue)

            background = util.normalizeHeatmap(background)


        heatmaps.append(background)

        return np.array(heatmaps).astype(np.float32)


    @staticmethod
    def Load3DJointPos(dataset):
        joints = dataset.worldTransforms

        camPose = util.GetNormCameraPose()

        camIntrMatrix = dataset.rgbFrame.cameraInfo.K
        jointMaps = []

        for i in range(0, joints.__len__()):
            jointMap = np.zeros(shape=config.inputRes)

            #just throw raw trans and rot into the system - we will see
            jointVals = []
            jointVals.extend(joints[i].trans.toList())
            jointVals.extend(joints[i].rotation.quat.toList())

            normPosition = util.ProjectToNormSpace(camPose, camIntrMatrix, joints[i].toList())

            for j in range(0, 7):
                if normPosition[0] < config.inputRes[0] and normPosition[1] < config.inputRes[1]:
                    jointMap[normPosition[0], normPosition[1]] = jointVals[j]

            jointMaps.append(np.array(jointMap))

        jointMaps = np.transpose(jointMaps, (1, 2, 0))
        return np.array(jointMaps)

    @staticmethod
    def ResizeDatasetNumpy(dataset: Dataset, image: np.ndarray, size=config.inputRes, crop=config.cropInputToBB):

        if crop:
            myImage = dataset.rgbFrame.boundingBox.CropImage(image)

            if dataset.backgroundMask is not None:
                dataset.backgroundMask = dataset.rgbFrame.boundingBox.CropImage(dataset.backgroundMask)
        else:
            myImage = image

        h, w, _ = myImage.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        dataset.rgbFrame.usedPadding = pad
        resizedBackground = None
        if config.keepAspectRatio:
            # Add padding
            #myImage = np.pad(myImage, pad, 'constant', constant_values=127.5)
            myImage = np.pad(myImage, pad, 'constant', constant_values=255 * config.paddingValue)

            if dataset.backgroundMask is not None:
                backgroundPad = ((pad1, pad2), (0, 0)) if h <= w else ((0, 0), (pad1, pad2))
                resizedBackground = np.pad(dataset.backgroundMask, backgroundPad, 'constant', constant_values=0.0)

        if myImage.shape[0] == 0 or myImage.shape[1] == 0:
            #this means no robot at all is on the image, so it does not matter what we use as input
            myImage = image

        resizedImage = cv2.resize(myImage, (size[1], size[0]))

        fv = size[0] / myImage.shape[0]
        fu = size[1] / myImage.shape[1]

        if dataset.backgroundMask is not None:
            resizedBackground = resize(resizedBackground, (size[0], size[1]), mode='reflect', anti_aliasing=False)

        if dataset.yoloData is not None:
            dataset.yoloData.resizedBoundingBoxes = []
            for bb in dataset.yoloData.boundingBoxes:
                resizedBB = copy.deepcopy(bb)
                resizedBB = resizedBB.AddPadding(pad[1][0], pad[0][0])
                resizedBB = resizedBB.ScaleCoordiantes(Vector2D(fv, fu))
                dataset.yoloData.resizedBoundingBoxes.append(resizedBB)

        if dataset.rgbFrame.projectedJoints is not None:
            dataset.rgbFrame.resizedReprojectedPoints.clear()
            dataset.rgbFrame.resizedReprojectedGT.clear()
            for pose in dataset.rgbFrame.projectedJoints:
                if pose[0] == -1 or pose[1] == -1:
                    remapedPoseGT = pose
                    remapedPosePadded = pose
                else:
                    if h <= w:
                        remapPadding = (0, pad1)
                    else:
                        remapPadding = (pad1, 0)

                    if crop:
                        remapedPoseGT = [pose[0] - dataset.rgbFrame.boundingBox.x1,
                                       pose[1] - dataset.rgbFrame.boundingBox.y1]
                        remapedPosePadded = [
                            (pose[0] - dataset.rgbFrame.boundingBox.x1) * fu + remapPadding[0] * fu,
                            (pose[1] - dataset.rgbFrame.boundingBox.y1) * fv + remapPadding[1] * fv
                        ]
                    else:
                        remapedPoseGT = [pose[0], pose[1]]
                        remapedPosePadded = [(pose[0] + remapPadding[0]) * fu, (pose[1] + remapPadding[1]) * fv]

                dataset.rgbFrame.resizedReprojectedPoints.append(Pose2D.fromData(x=remapedPosePadded[0],
                                                                                 y=remapedPosePadded[1],
                                                                                 angle=0.0))

                dataset.rgbFrame.resizedReprojectedGT.append(Pose2D.fromData(x=remapedPoseGT[0],
                                                                             y=remapedPoseGT[1],
                                                                             angle=0.0))


        dataset.backgroundHeatmap = resizedBackground
        return resizedImage
