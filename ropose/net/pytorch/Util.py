import sys, os
from typing import List, Dict, Tuple
from guthoms_helpers.base_types.Pose2D import Pose2D
from ropose_dataset_tools.DataClasses.Dataset import Dataset
from guthoms_helpers.base_types.BoundingBox2D import BoundingBox2D as BoundingBox
from guthoms_helpers.base_types.Vector2D import Vector2D
from guthoms_helpers.base_types.Vector3D import Vector3D
from guthoms_helpers.common_helpers.RandomHelper import RandomHelper
from ropose_dataset_tools.DataClasses.DetectionTypes.YoloDetection import YoloDetection
from ropose_dataset_tools.DataClasses.DetectionTypes.KeypointDetection import KeypointDetection
from ropose_dataset_tools.Augmentation.Augmenter import Augmenter
import torch
import torch.nn
from PIL import Image
import numpy as np
from random import randint
import ropose.pytorch_config as config
import copy
import math
import cv2
import io
import matplotlib.cm as mplColormaps
from scipy.ndimage import gaussian_filter

from ropose.thirdparty.yolov3.utils.utils import non_max_suppression

from skimage.transform import resize
from scipy.ndimage import zoom
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import transforms3d as transformation
from skimage import io
import random
import io as stdIO
import time

class Timer:
    startTime = time.time()
    text = ""

    def Start(self, text=""):
        self.text = text
        self.startTime = time.time()
        return

    def Stop(self, printLine = True):
        elapsed = time.time() - self.startTime
        if printLine:
            print(self.text + " Elapsedtime: " + str(elapsed) + " sec, FPS: " + str(1/elapsed))

        return elapsed

class Util:

    @staticmethod
    def ToCameraCoordinates(camPose, pose):

        translation = camPose[0]
        translation = np.reshape(translation, (3, 1))
        rotMat = transformation.quaternions.quat2mat(camPose[1])

        mat = np.append(rotMat, translation, axis=1)
        mat = np.append(mat, np.transpose(np.array([[0.0], [0.0], [0.0], [1.0]])), axis=0)

        position = np.reshape(pose[0], (3, 1))
        position = np.append(position, np.ones((1, 1)), axis=0)
        p_cam = np.matrix(mat) * np.matrix(position)

        return p_cam

    @staticmethod
    def ProjectToNormSpace(camPose, camIntrMatrix, pose):
        newPose = Util.ToCameraCoordinates(camPose, pose)

        temp = np.zeros((3, 1), dtype=float)
        camIntrMatrix = np.append(camIntrMatrix, temp, axis=1)

        pose_projec = camIntrMatrix * newPose

        xRes = camIntrMatrix[0, 0] * 2
        yRes = camIntrMatrix[1, 1] * 2

        ret = []
        ret.append(int((pose_projec[0] / pose_projec[2])/xRes * config.inputRes[0]))
        ret.append(int((pose_projec[0] / pose_projec[2])/yRes * config.inputRes[1]))

        return ret

    @staticmethod
    def GetNormCameraPose():

        camPosition = np.array([config.robotMaxWorkSpace, config.robotMaxWorkSpace, config.robotMaxWorkSpace])

        basePosition = np.array([0.0, 0.0, 0.0])

        viewVector = camPosition - basePosition

        viewVector = viewVector / np.linalg.norm(viewVector)

        dot = np.dot(viewVector, basePosition)

        angle = np.math.acos(dot)

        cross = np.cross(viewVector, basePosition)

        if cross[1] >= 0.0:
            angle *= -1

        ret = []
        ret.append(camPosition)
        ret.append(transformation.quaternions.axangle2quat([0.0, 0.0, 1.0], angle))

        return ret


    @staticmethod
    def Load2DJointInput(dataset: Dataset) -> np.array:
        ret = []

        joints = dataset.rgbFrame.projectedJoints
        normVec = Vector2D(dataset.rgbFrame.cameraInfo.width, dataset.rgbFrame.cameraInfo.height)
        for joint in joints:
            trans = joint.trans.NormalizedWithMax(normVec)
            ret.append(trans.toNp())

        return np.array(ret)

    @staticmethod
    def Displace2DJoints(joints: List[Pose2D], min: int = 0, max: int = 10) -> np.array:

        for i in range(0, len(joints)):
            joints[i].trans.x += RandomHelper.RandomInt(min, max)
            joints[i].trans.y += RandomHelper.RandomInt(min, max)

        return joints

    @staticmethod
    def Load3DJointGT(dataset: Dataset) -> np.array:
        ret = []

        joints = dataset.rgbFrame.transforms

        datasetMaxValue = 11.265220208599654
        normVec = Vector3D(datasetMaxValue, datasetMaxValue, datasetMaxValue)
        for joint in joints:
            #todo maybe introduce more normalization
            #trans = joint.trans
            trans = joint.trans.NormalizedWithMax(normVec)
            ret.append(trans.toNp())

        return np.array(ret)


    @staticmethod
    def Load3DJointLength(dataset: Dataset) -> np.array:
        ret = []

        joints = dataset.rgbFrame.transforms

        datasetMaxValue = 11.265220208599654
        normVec = Vector3D(datasetMaxValue, datasetMaxValue, datasetMaxValue)
        for i in range(1, joints.__len__()):
            # todo maybe introduce more normalization
            # trans = joint.trans
            diff = joints[i-1].trans.Distance(joints[i].trans)
            diff /= datasetMaxValue
            ret.append(diff)

        return np.array(ret)
    @staticmethod
    def Load3DCamPose(dataset: Dataset) -> np.array:
        ret = []

        sensorPose = dataset.rgbFrame.sensorPose

        yCam = sensorPose.trans.toList()
        rotation = sensorPose.rotation.axiAngles.toList()

        datasetMaxValue = 11.265220208599654
        normVec = Vector3D(datasetMaxValue, datasetMaxValue, datasetMaxValue)

        #todo maybe introduce more normalization
        sensorPose.trans = sensorPose.trans.NormalizedWithMax(normVec)
        ret.extend(sensorPose.trans.toList())

        rotation = sensorPose.rotation.axiAngles.toList()
        ret.extend(rotation[0])
        ret.append(rotation[1])

        return np.array(ret)

    @staticmethod
    def DrawOverlappingHeatmaps(image: np.array, heatmaps: np.array, boundingBox: BoundingBox, alpha: float= 0.3):
        resizedHeatmaps = boundingBox.ResizeImageToBoundingBox(heatmaps)
        return boundingBox.DrawImageInBoundingBox(image, resizedHeatmaps, alpha=alpha)

    @staticmethod
    def UnpadImage(image: np.array, padding: np.array):
        ret = image[
              padding[0][0]:image.shape[0]-padding[0][1],
              padding[1][0]:image.shape[1]-padding[1][1],
              padding[2][0]:image.shape[2]-padding[2][1],
              ]

        return ret

    @staticmethod
    def ToUintImage(img: np.ndarray) -> np.array:
        img *= 255
        img = img.astype(np.uint8)
        return img

    @staticmethod
    def ToFloat64Image(img: np.ndarray) -> np.array:
        img = img.astype(np.float64)
        img /= 255
        return img

    @staticmethod
    def OverlayHeatmaps(heatmaps):
        heatmaps = np.array(heatmaps)
        resHeatmap = np.amax(heatmaps, axis=0)
        #resHeatmap = Util.normalizeHeatmap(resHeatmap)

        return resHeatmap

    @staticmethod
    def CreateOverlappingImage(image, heatmaps, upsampling=True):

        if upsampling:
            upsampledHeatmap = Util.UpsampleHeatmapsThreaded(heatmaps, image.shape[0:3])
            heatmaps = upsampledHeatmap

        #Define color map
        cmap = mplColormaps.get_cmap('rainbow')
        cmap._init()
        alphas = np.linspace(0, 1.0, cmap.N + 3)
        cmap._lut[:, -1] = alphas

        #create figure
        fig = plt.figure(figsize=(10, 10))
        subPlot = fig.add_subplot(1, 1, 1)

        plt.imshow(image)
        for heatmap in heatmaps:
            heatmap = Util.normalizeHeatmap(heatmap)
            plt.imshow(heatmap, cmap=cmap, vmin=config.heatmapMinValue, vmax=config.heatmapMaxValue, alpha=0.8)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        plt.close()
        return image

    @staticmethod
    def StickImagesTogether(images, columnCount=2):

        count = images.__len__()
        oldWidth = images[0].width
        oldHeight = images[0].height
        width = images[0].width * columnCount

        if (columnCount > count):
            raise ValueError("columnCount can't be larger than the number of images!")

        height = images[0].height * math.ceil(count / columnCount)

        resImage = Image.new('RGB', (width, height))
        for i in range(0, count):
            posX = oldWidth * (i % columnCount)
            posY = oldHeight * math.floor(i / columnCount)
            resImage.paste(images[i], (posX, posY))

        return resImage

    @staticmethod
    def CreateStageCollection(dataset, neuralNet):
        testImage, gt = Util.LoadXYToArray(datasets=dataset)
        predictedHeatmaps = neuralNet.Predict(dataset)

        partHeatmaps = []
        backgroundHeatmaps = []

        for i in range(0, predictedHeatmaps.__len__()):
            partHeatmaps.append(Util.CreateOverlappingImage(testImage[0], predictedHeatmaps[i][0][0:6]))
            backgroundHeatmaps.append(Util.CreateOverlappingImage(testImage[0], predictedHeatmaps[i][0][6:7]))

        partImage = Util.StickImagesTogether(partHeatmaps, columnCount=partHeatmaps.__len__())
        backgroundImage = Util.StickImagesTogether(backgroundHeatmaps, columnCount=backgroundHeatmaps.__len__())

        return partImage, backgroundImage


    @staticmethod
    def CreateImageCollection(datasets, neuralNet, includeGT=False, columnCount=2):

        #create info dict
        resultInformation = {}
        resultInformation["testImages"] = []
        resultInformation["groundTruths"] = []
        resultInformation["predictedHeatmaps"] = []

        partHeatmaps = []
        backgroundHeatmaps = []

        for dataset in datasets:
            testImage, gt = Util.LoadXYToArray(datasets=dataset)
            predictedHeatmaps = neuralNet.Predict(dataset)

            #extend lists
            resultInformation["testImages"].extend(testImage)
            resultInformation["groundTruths"].extend(gt)
            resultInformation["predictedHeatmaps"].extend([predictedHeatmaps])

            # for outputs in predictedHeatmaps:
            if includeGT:
                partHeatmaps.append(Util.CreateOverlappingImage(testImage[0], gt[0][0:-1]))
                backgroundHeatmaps.append(Util.CreateOverlappingImage(testImage[0], [gt[0][-1]]))

            outputNr = predictedHeatmaps.__len__() - 1
            partHeatmaps.append(Util.CreateOverlappingImage(testImage[0], predictedHeatmaps[outputNr][0][0:-1]))
            backgroundHeatmaps.append(Util.CreateOverlappingImage(testImage[0], [predictedHeatmaps[outputNr][0][-1]]))

        partImage = Util.StickImagesTogether(partHeatmaps, columnCount=columnCount)
        backgroundImage = Util.StickImagesTogether(backgroundHeatmaps, columnCount=columnCount)

        return partImage, backgroundImage, resultInformation

    @staticmethod
    def SplitDataset(dataset: List['Dataset'], trainFactor=70, testFactor=10, random=False,
                     checkValidity=True) -> Tuple[List['Dataset'], List['Dataset'], List['Dataset']]:

        trainSet = []
        testSet = []
        validationSet = []

        setLength = dataset.__len__()

        if checkValidity:
            for set in dataset:
                if not set.valid:
                    dataset.remove(set)

        if not random:
            trainCount = int(setLength/100 * trainFactor)
            testCount = int(setLength/100 * testFactor)

            startCount = 0
            trainSet = dataset[startCount:trainCount]

            startCount = trainCount + 1
            testSet = dataset[startCount:startCount + testCount]

            startCount = startCount + testCount + 1
            validationSet = dataset[startCount:dataset.__len__()-1]

        else:
            #copy dataset for splitting
            tempDataset = copy.deepcopy(dataset)

            while trainSet.__len__() < int(setLength/100 * trainFactor):
                index = randint(0, tempDataset.__len__() - 1)
                trainSet.append(tempDataset[index])
                tempDataset.pop(index)

            #create test set
            while testSet.__len__() < int(setLength / 100 * testFactor):
                index = randint(0, tempDataset.__len__() - 1)
                testSet.append(tempDataset[index])
                tempDataset.pop(index)

            for dataset in tempDataset:
                validationSet.append(dataset)

        return trainSet, testSet, validationSet

    @staticmethod
    def ImageToArray(image):
        array = np.array(image)
        return array

    @staticmethod
    def ToUintImage(img: np.ndarray) -> np.array:
        img *= 255
        img = img.astype(np.uint8)
        return img

    @staticmethod
    def ToFloat64Image(img: np.ndarray) -> np.array:
        img = img.astype(np.float64)
        img /= 255
        return img

    @staticmethod
    def PrepareRawInput(image, targetSize=config.inputRes, printTestTimers=False):
        timer = Timer()

        timer.Start("Calculate Ratios")
        h, w, _ = image.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        timer.Stop(printTestTimers)

        if config.keepAspectRatio:
            # Add padding
            timer.Start("Padding")
            image = np.pad(image, pad, 'constant', constant_values=255 * config.paddingValue)
            timer.Stop(printTestTimers)


        timer.Start("Image Resizing OPENCV")
        resizedImage = cv2.resize(image, (targetSize[1], targetSize[0]))
        timer.Stop(printTestTimers)

        resizeFactor = [image.shape[0] / targetSize[0], image.shape[1] / targetSize[1]]

        return resizedImage, pad, resizeFactor

    @staticmethod
    def RestoreKeyPointsUpsampled(keypoints, paddings=None, boundingBoxes: List[BoundingBox]=None):
        ret = []
        counter = 0
        for batch in keypoints:
            batchPoints = []
            for keypoint in batch:
                if keypoint[0] != -1 and keypoint[1] != -1:
                    if paddings is not None:
                        # substract padding from input creation
                        keypoint[0] -= paddings[counter][1][0]
                        keypoint[1] -= paddings[counter][0][0]
                    if boundingBoxes is not None:
                        keypoint[0] += boundingBoxes[counter].x1
                        keypoint[1] += boundingBoxes[counter].y1
                batchPoints.append(keypoint)

            counter += 1
            ret.append(batchPoints)

        return np.array(ret)

    @staticmethod
    def RestoreKeyPoints(keypoints, resizeFactor, paddings=None, outputSize=config.outputRes, targetSize=config.inputRes,
                         boundingBoxes: List[BoundingBox]=None):

        factor = ((float(targetSize[0]) / float(outputSize[0]),
                   float(targetSize[1]) / float(outputSize[1])
                   ))

        ret = []
        counter = 0
        for batch in keypoints:
            batchPoints = []
            for keypoint in batch:
                if keypoint[0] != -1 and keypoint[1] != -1:
                    # factor restores input size resizeFactor the origin size
                    keypoint[0] *= factor[1] * resizeFactor[counter][1]
                    keypoint[1] *= factor[0] * resizeFactor[counter][0]
                    if config.keepAspectRatio:
                        #substract padding from input creation
                        keypoint[0] -= paddings[counter][1][0]
                        keypoint[1] -= paddings[counter][0][0]
                    if boundingBoxes is not None:
                        keypoint[0] += boundingBoxes[counter].x1
                        keypoint[1] += boundingBoxes[counter].y1
                batchPoints.append(keypoint)

            counter += 1
            ret.append(batchPoints)

        return np.array(ret)

    @staticmethod
    def RestoreKeyPointsYolo(keypoints, resizeFactor, pad=None, detections: List[YoloDetection] = None):

        if config.includeUpsampling:
            factor = 1.0
        else:
            factor = config.downsampleFactor

        ret = []
        counter = 0
        for batch in keypoints:
            batchPoints = []
            for keypoint in batch:
                if keypoint[0] != -1 and keypoint[1] != -1:
                    # factor restores input size resizeFactor the origin size
                    keypoint[0] *= factor * resizeFactor[counter][0]
                    keypoint[1] *= factor * resizeFactor[counter][1]

                    if config.keepAspectRatio:
                        #substract padding from input creation
                        keypoint[0] -= pad[counter][1][0]
                        keypoint[1] -= pad[counter][0][0]

                    if detections is not None:
                        keypoint[0] += detections[counter].boundingBox.x1
                        keypoint[1] += detections[counter].boundingBox.y1

                batchPoints.append(keypoint)

            counter += 1
            ret.append(batchPoints)

        return np.array(ret)

    @staticmethod
    def RestoreKeyPointsUpsampledYolo(keypoints, pad=None, detection: YoloDetection = None):
        ret = []
        for batch in keypoints:
            for keypoint in batch:
                if keypoint[0] != -1 and keypoint[1] != -1:
                    if config.keepAspectRatio:
                        #substract padding from input creation
                        keypoint[0] -= pad[1][0]
                        keypoint[1] -= pad[0][0]
                    if detection is not None:
                        keypoint[0] += detection.boundingBox.x1
                        keypoint[1] += detection.boundingBox.y1

                ret.append(keypoint)

        return np.array(ret)

    @staticmethod
    def RemovePaddingFromImage(image: np.array, padding: np.array) -> np.array:
        ret = image
        return ret


    @staticmethod
    def ExtractKeypoints(heatmaps):

        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]
        width = heatmaps.shape[3]
        heatmaps = heatmaps.cpu().numpy()
        heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals

    @staticmethod
    def normalizeHeatmap(heatmap, maxValue = config.heatmapMaxValue, minValue = config.heatmapMinValue):
        max = np.max(heatmap)
        min = np.min(heatmap)

        range = max - min
        if range is not np.nan and not range <= 0.0:
            newRange = maxValue - minValue
            normalized = (heatmap - min) / range

            normalized = normalized * newRange - abs(minValue)
        else:
            normalized = heatmap

        return normalized

    @staticmethod
    def CreateZeroHeatmaps(layers):
        heatmaps = torch.zeros((layers, config.outputRes[0], config.outputRes[1]))
        return heatmaps

    @staticmethod
    def CropImage(image: np.ndarray,  bb: BoundingBox):
        return bb.CropImage(image)

    @staticmethod
    def ResizeImage(image: np.ndarray, size=config.inputRes):
        fu = size[0] / image.shape[1]
        fv = size[1] / image.shape[0]

        if config.keepAspectRatio:
            fu = fv = min(fu, fv)

        resizedImage = cv2.resize(src=image, dsize=None, fx=fu, fy=fv)

        if config.keepAspectRatio:
            paddingColor = [0, 0, 0]
            side = (size[0] - resizedImage.shape[0])
            up = (size[1] - resizedImage.shape[1])

            resizedImage = cv2.copyMakeBorder(resizedImage, 0, side, 0, up, cv2.BORDER_CONSTANT, value=paddingColor)


        return resizedImage

    @staticmethod
    def ShowHeatmaps(heatmaps, split=True):

        if split:
            cmap = mplColormaps.get_cmap('rainbow')
            fig = plt.figure(figsize=(10, 10))

            for i in range(0, heatmaps.__len__()):
                subPlot = fig.add_subplot(3, 3, i+1)
                subPlot.set_title("Heatmap " + str(i))
                heatmap = np.array(heatmaps[i].cpu())

                plt.pcolormesh(heatmap, cmap=cmap, vmin=np.min(heatmap), vmax=np.max(heatmap))

        plt.tight_layout()
        plt.show()
        return

    @staticmethod
    def ShowInput(input):

        plt.imshow(input)
        plt.tight_layout()
        plt.show()
        return

    @staticmethod
    def ShowDebugImage(image, showTime: int=1000, name="debug image"):
        cv2.imshow(name, image)
        cv2.waitKey(showTime)
        return

    @staticmethod
    def PlotRoPoseSet(dataset, neuralNet, split=True, overLap=False, upsampling=True, appendGT=True, preprocessFunction=None,
                augmentation=False):

        if augmentation:
            x, x_joint, y = Util.LoadAugDataset(dataset)
        else:
            x, x_joint, y = Util.LoadXY(dataset)

        image = x
        image_input = image.transpose((2, 0, 1))
        image_input = preprocessFunction(torch.from_numpy(image_input).float())
        image_input = image_input.unsqueeze(0)
        joints = np.array([x_joint])
        gt = y

        #Util.ShowInput(image)

        if config.use3DJointInput:
            array = neuralNet.Predict(image_input.to(neuralNet.net.device),
                                      torch.from_numpy(joints).to(neuralNet.device)).cpu().numpy()
        else:
            array = neuralNet.Predict(image_input.to(neuralNet.device)).cpu().numpy()

        #Util.ShowHeatmaps(array)

        array = array[-1]

        perRow = int((array.shape[0]+4) / 4)
        figData = [4, perRow]

        if upsampling:
            upsampledHeatmap = (Util.UpsampleHeatmapsThreaded(array, image.shape[0:3]))
            upsampledGT = (Util.UpsampleHeatmapsThreaded(gt, image.shape[0:3]))

            array = upsampledHeatmap
            gt = upsampledGT

        if split:
            fig = plt.figure(figsize=(10, 10))

            # cmap = mplColors.lin.from_list('heatmap', ['blue', 'red'], 256)
            cmap = mplColormaps.get_cmap('rainbow')
            cmap._init()
            alphas = np.linspace(0, 1.0, cmap.N + 3)
            cmap._lut[:, -1] = alphas

            if overLap:
                subPlot = fig.add_subplot(1, 1, 1)

                plt.imshow(image)
                for arr in array:
                    max = np.max(arr)
                    min = np.min(arr)
                    # plt.contourf(arr, cmap=cmap, vmin=config.heatmapMinValue, vmax=np.max(arr))
                    plt.imshow(arr, cmap=cmap, vmin=min, vmax=max, alpha=0.5)

            else:
                subPlot = fig.add_subplot(figData[0], figData[0], 1)
                plt.imshow(image)

                # plot predicted Background
                subPlot = fig.add_subplot(figData[0], figData[0], 2)
                subPlot.set_title("Pred. Background")
                heatmap = array[-1]

                if upsampling:
                    plt.imshow(image, alpha=0.7)
                    plt.pcolormesh(heatmap, cmap=cmap, vmin=np.min(heatmap), vmax=np.max(heatmap), alpha=0.2)
                else:
                    plt.pcolormesh(heatmap, cmap=cmap, vmin=np.min(heatmap), vmax=np.max(heatmap))

                # create GT image
                if appendGT:
                    subPlot = fig.add_subplot(figData[0], figData[0], 3)
                    subPlot.set_title("GT parts")
                    together = Util.OverlayHeatmaps(gt[0:-1])
                    together = Util.normalizeHeatmap(together)
                    if upsampling:
                        plt.imshow(image, alpha=0.7)
                        plt.pcolormesh(together, cmap=cmap, vmin=np.min(together), vmax=np.max(together), alpha=0.2)
                    else:
                        plt.pcolormesh(together, cmap=cmap, vmin=np.min(together), vmax=np.max(together))

                    subPlot = fig.add_subplot(figData[0], figData[0], 4)
                    subPlot.set_title("GT Background")
                    gtBackground = gt[-1]

                    plt.gca().invert_yaxis()
                    plt.pcolormesh(gtBackground, cmap=cmap, vmin=np.min(gtBackground), vmax=np.max(gtBackground))

                #can plot max af 16 imges
                for i in range(np.min([array.__len__() - 1, 11])):
                    subPlot = fig.add_subplot(figData[0], figData[0], i + 5)

                    subPlot.set_title("Pred. Joint " + str(i))
                    maxVal = np.max(array[i])
                    # subPlot.set_title(str(i) + " MaxVal: " + str(maxVal))
                    if upsampling:
                        plt.imshow(image, alpha=0.7)
                        plt.pcolormesh(array[i], cmap=cmap, vmin=np.min(array[i]), vmax=maxVal, alpha=0.2)
                    else:
                        plt.pcolormesh(array[i], cmap=cmap, vmin=np.min(array[i]), vmax=maxVal)

        else:
            fig = plt.figure(figsize=(10, 10))
            # plot raw image first
            subPlot = fig.add_subplot(1, 2, 1)
            plt.imshow(image)
            subPlot = fig.add_subplot(1, 2, 2)
            plt.imshow(array)

        plt.tight_layout()
        buf = stdIO.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        plt.close()
        return image, buf

    @staticmethod
    def PlotColPoseSet(dataset, neuralNet, split=True, overLap=False, upsampling=True, appendGT=True,
                      preprocessFunction=None,
                      augmentation=False):

        if augmentation:
            x, x_joint, y = Util.LoadAugDataset(dataset)
        else:
            x, x_joint, y = Util.LoadXY(dataset)

        rawImage = x
        image_input = rawImage.transpose((2, 0, 1))
        image_input = preprocessFunction(torch.from_numpy(image_input).float())
        image_input = image_input.unsqueeze(0)
        joints = np.array([x_joint])
        gt = y

        # Util.ShowInput(image)

        if config.use3DJointInput:
            outputs = neuralNet.Predict(image_input.to(neuralNet.device),
                                      torch.from_numpy(joints).to(neuralNet.device)).cpu().numpy()
        else:
            outputs = neuralNet.Predict(image_input.to(neuralNet.device))

        images = []
        bufs = []
        for output in outputs:
            output = output.cpu().numpy()
            for array in output:
                perRow = int((array.shape[0] + 4) / 4)
                figData = [4, perRow]

                if upsampling:
                    upsampledHeatmap = (Util.UpsampleHeatmapsThreaded(array, rawImage.shape[0:3]))
                    upsampledGT = (Util.UpsampleHeatmapsThreaded(gt, rawImage.shape[0:3]))

                    array = upsampledHeatmap
                    gt = upsampledGT

                if split:
                    fig = plt.figure(figsize=(10, 10))

                    # cmap = mplColors.lin.from_list('heatmap', ['blue', 'red'], 256)
                    cmap = mplColormaps.get_cmap('rainbow')
                    cmap._init()
                    alphas = np.linspace(0, 1.0, cmap.N + 3)
                    cmap._lut[:, -1] = alphas

                    if overLap:
                        subPlot = fig.add_subplot(1, 1, 1)

                        plt.imshow(rawImage)
                        for arr in array:
                            max = np.max(arr)
                            min = np.min(arr)
                            # plt.contourf(arr, cmap=cmap, vmin=config.heatmapMinValue, vmax=np.max(arr))
                            plt.imshow(arr, cmap=cmap, vmin=min, vmax=max, alpha=0.5)

                    else:
                        subPlot = fig.add_subplot(figData[0], figData[0], 1)
                        plt.imshow(rawImage)

                        # plot predicted Background
                        subPlot = fig.add_subplot(figData[0], figData[0], 2)
                        subPlot.set_title("Pred. Background")
                        heatmap = array[-1]

                        if upsampling:
                            plt.imshow(rawImage, alpha=0.7)
                            plt.pcolormesh(heatmap, cmap=cmap, vmin=np.min(heatmap), vmax=np.max(heatmap), alpha=0.2)
                        else:
                            plt.pcolormesh(heatmap, cmap=cmap, vmin=np.min(heatmap), vmax=np.max(heatmap))

                        # create GT image
                        if appendGT:
                            subPlot = fig.add_subplot(figData[0], figData[0], 3)
                            subPlot.set_title("GT parts")
                            together = Util.OverlayHeatmaps(gt[0:-1])
                            together = Util.normalizeHeatmap(together)
                            if upsampling:
                                plt.imshow(rawImage, alpha=0.7)
                                plt.pcolormesh(together, cmap=cmap, vmin=np.min(together), vmax=np.max(together), alpha=0.2)
                            else:
                                plt.pcolormesh(together, cmap=cmap, vmin=np.min(together), vmax=np.max(together))

                            subPlot = fig.add_subplot(figData[0], figData[0], 4)
                            subPlot.set_title("GT Background")
                            gtBackground = gt[-1]
                            plt.gca().invert_yaxis()
                            plt.pcolormesh(gtBackground, cmap=cmap, vmin=np.min(gtBackground), vmax=np.max(gtBackground))

                        # can plot max af 16 imges
                        for i in range(np.min([array.__len__() - 1, 11])):
                            subPlot = fig.add_subplot(figData[0], figData[0], i + 5)

                            subPlot.set_title("Pred. Joint " + str(i))
                            maxVal = np.max(array[i])
                            # subPlot.set_title(str(i) + " MaxVal: " + str(maxVal))
                            if upsampling:
                                plt.imshow(rawImage, alpha=0.7)
                                plt.pcolormesh(array[i], cmap=cmap, vmin=np.min(array[i]), vmax=maxVal, alpha=0.2)
                            else:
                                plt.pcolormesh(array[i], cmap=cmap, vmin=np.min(array[i]), vmax=maxVal)

                else:
                    fig = plt.figure(figsize=(10, 10))
                    # plot raw image first
                    subPlot = fig.add_subplot(1, 2, 1)
                    plt.imshow(rawImage)
                    subPlot = fig.add_subplot(1, 2, 2)
                    plt.imshow(array)

                plt.tight_layout()
                buf = stdIO.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                image = Image.open(buf)
                images.append(image)
                bufs.append(buf)
                plt.close()

        return images, bufs

    @staticmethod
    def SwapDatasetData(dataset: List['Dataset']):
        np.random.shuffle(dataset)
        return dataset

    @staticmethod
    def DrawYoloBBGT(img: np.array, detections: np.array):
        # Bounding-box colors
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (config.yolo_InputSize[0] / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (config.yolo_InputSize[1] / max(img.shape))
        # Image height and width after padding is removed
        unpad_h = config.yolo_InputSize[0] - pad_y
        unpad_w = config.yolo_InputSize[1] - pad_x

        bbox_color = colors[0]
        imgW, imgH, _ = img.shape
        # Create plot
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        for _, cls_pred, x, y, w, h in detections:
            if x == 0 and y == 0 and w == 0 and h == 0:
                continue

            # Rescale coordinates to original dimensions
            box_h = h * imgH
            box_w = w * imgW
            y1 = y * imgH - box_h/2
            x1 = x * imgW - box_w/2

            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                     edgecolor=bbox_color,
                                     facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(x1, y1, s=config.yolo_Classes[int(cls_pred+1)]["name"], color='white', verticalalignment='top',
                     bbox={'color': bbox_color, 'pad': 0})

        # Save generated image with detections
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        return plt

    @staticmethod
    def RestoreBoundingBoxes(detections: np.array, padding, resizeFactor):
        # Iterate through images and save plot of detections
        retDetections = []
        #scaleFactor = [resizeFactor[0], resizeFactor[1]]

        for detection in detections:
            if detection is not None:
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    boundingBox = BoundingBox(x1.cpu().numpy(), y1.cpu().numpy(), x2.cpu().numpy(), y2.cpu().numpy())
                    boundingBox = boundingBox.ScaleBB(resizeFactor[0], resizeFactor[1])
                    boundingBox = boundingBox.SubstractPadding(padding[1][0], padding[0][0])
                    retDetections.append((boundingBox, cls_pred.cpu().numpy(), cls_conf.cpu().numpy()))

        return retDetections

    @staticmethod
    def DrawBBs(image: np.array, detections: List[YoloDetection]):

        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for detection in detections:
            detection.boundingBox.AddPatch(plt, ax, config.yolo_Classes[int(detection.predictedClass)+1]["name"])

        return plt

    @staticmethod
    def DrawYoloBB(img: np.array, detections: np.array):
        #taken/modified from https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/detect.py
        # Bounding-box colors
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        # Iterate through images and save plot of detections
        for detection in detections:
            # Create plot
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            # The amount of padding that was added
            pad_x = max(img.shape[0] - img.shape[1], 0) * (config.yolo_InputSize[0] / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) * (config.yolo_InputSize[1] / max(img.shape))
            # Image height and width after padding is removed
            unpad_h = config.yolo_InputSize[0] - pad_y
            unpad_w = config.yolo_InputSize[1] - pad_x

            # Draw bounding boxes and labels of detections
            if detection is not None:
                unique_labels = detection[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    print('\t+ Label: %s, Conf: %.5f' % (config.yolo_Classes[int(cls_pred)]["name"], cls_conf.item()))

                    # Rescale coordinates to original dimensions
                    box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                    box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                    y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                    x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                             edgecolor=color,
                                             facecolor='none')
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(x1, y1, s=config.yolo_Classes[int(cls_pred)], color='white', verticalalignment='top',
                             bbox={'color': color, 'pad': 0})

            # Save generated image with detections
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            return plt

    @staticmethod
    def UpsampleHeatmap(heatmap, targetSize=config.inputRes, normalize=False, index=None):

        #perform upsampling order=1 -> biliniar order=3-> cubic
        factor = ((float(targetSize[0])/float(heatmap.shape[0]), float(targetSize[1])/float(heatmap.shape[1])))
        upsampled = zoom(heatmap, factor, order=3)

        if normalize:
            upsampled = Util.normalizeHeatmap(upsampled)

        return upsampled, index

    @staticmethod
    def UpsampleHeatmaps(array, targetSize=config.inputRes):
        heatmaps = []
        for heatmap in array:
            heatmaps.append(Util.UpsampleHeatmap(heatmap, targetSize=targetSize))

        return heatmaps

    @staticmethod
    def UpsampleHeatmapsGPU(heatmaps, targetSize=config.inputRes, resizeFactor=np.array([1.0, 1.0])):

        factor = ((float(targetSize[0]) / float(heatmaps.shape[2]) * resizeFactor[0],
                   (float(targetSize[1]) / float(heatmaps.shape[3]) * resizeFactor[1])))

        upsampler = torch.nn.Upsample(size=targetSize, mode="bilinear").cuda()

        upsampled = upsampler(heatmaps)

        return upsampled

    @staticmethod
    def UpsampleBatchHeatmapsGPU(heatmaps, targetSizes=None, resizeFactors=None, method: str = "bicubic"):

        upsampled = []
        singleHeatmaps = torch.unbind(heatmaps, dim=0)
        batchsize = heatmaps.shape[0]

        if targetSizes is None:
            targetSizes = [config.inputRes for i in range(0, batchsize)]

        for i in range(0, batchsize):
            if resizeFactors is not None:
                size = (int(targetSizes[i][0] * resizeFactors[i][0]), int(targetSizes[i][1] * resizeFactors[i][1]))
            else:
                size = (int(targetSizes[i][0]), int(targetSizes[i][1]))

            #TODO: Find a way to create a multidimensional upsample maybe we can save some more time here.
            upsampler = torch.nn.Upsample(size=size, mode=method).cuda()
            singleUpsampled = upsampler(torch.unsqueeze(singleHeatmaps[i], dim=0))
            upsampled.append(torch.squeeze(singleUpsampled, 0))

        return upsampled

    @staticmethod
    def CombineHeatmapsGPU(heatmaps):

        combined = torch.sum(heatmaps, dim=1)

        return combined

    @staticmethod
    def UpsampleHeatmapsThreaded(array, targetSize=config.inputRes, normalize=False):
        heatmaps = [None]*(array.__len__())

        threadpool = ThreadPool(processes=array.__len__())
        async_results = []
        for i in range(0, array.__len__()):
            async_results.append(threadpool.apply_async(Util.UpsampleHeatmap, (array[i], targetSize, normalize, i)))

        for result in async_results:
            res = result.get()
            heatmaps[res[1]] = res[0]
        threadpool.close()
        return heatmaps


    @staticmethod
    def FindHeatmapPeaksThreaded(heatmaps):
        peaks = [None] * (heatmaps.__len__())

        threadpool = ThreadPool(processes=heatmaps.__len__())

        async_results = []
        for i in range(0, heatmaps.__len__()):
            # res, index = Util.DetectPeaks(heatmaps[i], i)
            async_results.append(threadpool.apply_async(Util.DetectPeaks, (heatmaps[i], i)))

        for result in async_results:
            res = result.get()
            peaks[res[1]] = res[0]

        threadpool.close()

        return peaks


    @staticmethod
    def GetBestPeak(peaks):
        npPeaks = np.array(peaks)[:, -1]

        bestIndex = np.argmax(npPeaks)

        return peaks[bestIndex]


    @staticmethod
    def DetectPeaks(heatmap, index):
        if np.max(heatmap) <= 0.0:
            return (np.nan, np.nan, 0.0), index

        heatmap = gaussian_filter(heatmap, sigma=3)

        map_left = np.zeros(heatmap.shape)
        map_left[1:, :] = heatmap[:-1, :]
        map_right = np.zeros(heatmap.shape)
        map_right[:-1, :] = heatmap[1:, :]
        map_up = np.zeros(heatmap.shape)
        map_up[:, 1:] = heatmap[:, :-1]
        map_down = np.zeros(heatmap.shape)
        map_down[:, :-1] = heatmap[:, 1:]

        is_local_peak_list = np.logical_and.reduce(
            (heatmap >= map_left, heatmap >= map_right,
             heatmap >= map_up, heatmap >= map_down,
             heatmap > config.rejectionTH)
        )

        peaks = list(zip(np.nonzero(is_local_peak_list)[1], np.nonzero(is_local_peak_list)[0]))
        peaks_with_score = [x + (heatmap[x[1], x[0]],) for x in peaks]

        # if no peak was found
        if peaks_with_score.__len__() == 0:
            return (np.nan, np.nan, np.max(heatmap)), index

        bestPeak = Util.GetBestPeak(peaks_with_score)

        return bestPeak, index

    @staticmethod
    def CalculatePixelDistance(pixelA, pixelB):
        distance = math.sqrt(pow(pixelA.x - pixelB.x, 2)
                  + (pow(pixelA.y - pixelB.y, 2)))
        return distance

    @staticmethod
    def FilterDetection(pred, probabilities, rejectionTH):
        poseKeypoints = []

        for batch in range(0, len(pred)):

            invisibleCounter = 0

            setPred = pred[batch]
            setProb = probabilities[batch]

            tempList = []
            for i in range(0, len(setPred)):
                if setProb[i] > rejectionTH:
                    tempList.append(Pose2D.fromData(setPred[i][0], setPred[i][1], 0.0, visible=True))
                else:
                    invisibleCounter += 1
                    tempList.append(Pose2D.fromData(setPred[i][0], setPred[i][1], 0.0, visible=False))

            if invisibleCounter <= tempList.__len__():
                #add only if at least ONE joint is visible
                poseKeypoints.append(tempList)

        return poseKeypoints

    @staticmethod
    def GetAbosluteKeypointDistances(pred: Pose2D, gt: Pose2D):
        distances = []
        for i in range(0, pred.__len__()):
            batchDistances = []
            for j in range(0, len(pred[i])):
                distance = np.power(gt[i][j][0]-pred[i][j][0], 2) + np.power(gt[i][j][1]-pred[i][j][1], 2)
                batchDistances.append(np.sqrt(distance))
            distances.append(batchDistances)
        return distances

    @staticmethod
    def GetAbsolutePartDistances(gt_heatmaps, pred_heatmaps):
        gt_upsampled = Util.UpsampleHeatmapsThreaded(gt_heatmaps, config.inputRes)
        gt_peaks = Util.FindHeatmapPeaksThreaded(gt_upsampled[0:6])
        pred_upsampled = Util.UpsampleHeatmapsThreaded(pred_heatmaps[2], config.inputRes)
        pred_peaks = Util.FindHeatmapPeaksThreaded(pred_upsampled[0:6])

        distances = []
        for i in range(0, pred_peaks.__len__()):
            distances.append(math.sqrt(pow(pred_peaks[i][0] - gt_peaks[i][0], 2)
                                       + pow(pred_peaks[i][1] - gt_peaks[i][1], 2)))

        return distances

    @staticmethod
    def SaveTestYoloCropedBoyes(dataset: Dataset, path: str="/home/thomas/ropose/yolo"):
        img = cv2.imread(dataset.rgbFrame.filePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        filename = os.path.join(path, "yoloData_example.jpg")
        cv2.imwrite(filename, img)
        #img = img / 255
        img = np.array(img)

        for i in range(0, dataset.yoloData.boundingBoxes.__len__()):
            bb = dataset.yoloData.boundingBoxes[i]
            filename = os.path.join(path, "yoloData_" + str(i) + ".jpg")
            cropedImg = bb.CropImage(img)
            cv2.imwrite(filename, cropedImg)
            print("Saved demo Box to: " + filename)

        pass

    @staticmethod
    def SaveYoloDirectInput(xRaw, yRaw, path: str=config.showExamplePath + "/"):

        x = copy.deepcopy(xRaw)
        y = copy.deepcopy(yRaw)

        filename = os.path.join(path, "yoloData_input_example.jpg")
        plt.figure()
        fig, ax = plt.subplots(1)

        xTrans = x.numpy().transpose((1, 2, 0))
        ax.imshow(xTrans)
        color = [0.0, 0.0, 0.0, 1.0]
        for i in range(0, y.shape[0]):
            gt = y[i]

            if gt[2] <= 0.0 or gt[3] <= 0.0:
                continue

            gt[2] *= x.shape[1]
            gt[3] *= x.shape[2]
            gt[4] *= x.shape[1]
            gt[5] *= x.shape[2]
            # Create a Rectangle patch
            px = gt[2] - gt[4]/2
            py = gt[3] - gt[5]/2
            bbox = patches.Rectangle((px, py), gt[4], gt[5], linewidth=2,
                                     edgecolor=color, facecolor='none')

            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(px, py, s=str(gt[1].cpu()), color='white', verticalalignment='top',
                     bbox={'color': color, 'pad': 0})



        plt.savefig(filename)


    @staticmethod
    def FilterYolo(detections: List[YoloDetection], humanClassNr: int = config.yolo_HumanClassNum,
                   roposeClassNr: int = config.yolo_RoposeClassNum, yoloInputSize=config.yolo_InputSize) \
            -> Tuple[List[YoloDetection], List[YoloDetection]]:

        humanDetections = []
        roposeDetections = []

        for detection in detections:
            detection.boundingBox = detection.boundingBox.ClipToShape(yoloInputSize)
            if detection.boundingBox.DiagLength() <= 0.0:
                #filter detections where the BB is "not existent" (= BB size = 0)
                continue

            if detection.predictedClass == humanClassNr:
                humanDetections.append(detection)
                continue

            if detection.predictedClass == roposeClassNr:
                roposeDetections.append(detection)
                continue

        return humanDetections, roposeDetections

    @staticmethod
    def YoloNonMaxSuppression(prediction, conf_thres=0.5, nms_thres=0.5):
        yoloDetections = []

        detections = non_max_suppression(prediction, conf_thres, nms_thres)

        #add our yolo detection conversion
        for i in range(0, detections.__len__()):
            if detections[i] is not None:
                for j in range(0, detections[i].shape[0]):
                    yoloDetections.append(YoloDetection.FromPredictionTensor(detections[i][j]))

        return yoloDetections

    @staticmethod
    def GetYoloClassName(detectedClassNr: int) -> str:

        name = config.yolo_Classes[detectedClassNr+1]["name"]
        return name

    @staticmethod
    def PredictYolo(rawFrame: np.array, yoloNet, augment: bool = False) -> Tuple[List[YoloDetection], Tuple, List]:
        x, padding, resizeFactor = Util.PrepareRawInput(rawFrame, config.yolo_InputSize, printTestTimers=False)

        if augment:
            raise Exception("implement with new Augmentor!")


        x = Util.ToFloat64Image(x)
        img = x.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        img = yoloNet.PreprocessInput(img)
        img = img.unsqueeze(0)

        pred = yoloNet.Predict(img)

        detections = Util.YoloNonMaxSuppression(pred, config.yolo_IgnoreThreshold, config.yolo_NonMaxSurThreshold)
        return detections, padding, resizeFactor

    @staticmethod
    def RestoreYoloBB(detection: YoloDetection, resizeFactor, padding) -> YoloDetection:

        detection.boundingBox = detection.boundingBox.ScaleCoordiantes(Vector2D(resizeFactor[1], resizeFactor[0]))
        detection.boundingBox = detection.boundingBox.SubstractPadding(padding[1][0], padding[0][0])
        detection.boundingBox = detection.boundingBox.ClipMin(0.0)

        return detection

    @staticmethod
    def ExtractYoloDetections(rawFrame, detections: List[YoloDetection]) -> List[np.array]:

        ret = []

        for detection in detections:
            ret.append(detection.boundingBox.CropImage(rawFrame))

        return ret


    @staticmethod
    def DrawKeypointDetections(image: np.array, detections: List[KeypointDetection], colors):
        resImage = image

        for detection in detections:
            #humans
            if detection.detectionsClass == 0:
                resImage = Util.DrawHumanPose(resImage, detection.keypoints, colors)
            #robots
            elif detection.detectionsClass == config.yolo_RoposeClassNum:
                resImage = Util.DrawPose(resImage, detection.keypoints, colors)
        return resImage


    @staticmethod
    def DrawPose(image, poses, colors, alpha=1.0):
        resImage = image
        for i in range(0, poses.__len__()):
            if not math.isnan(poses[i][0]) and not math.isnan(poses[i][1]) and poses[i].visible:
                # draw pose circles
                position = (int(poses[i][0]), int(poses[i][1]))
                resImage = cv2.circle(resImage, position, 5, colors[i], thickness=-1)

                # show score
                if position[0] == -1 or position[1] == -1:
                    textPosition = (10, 20 + i * 25)
                    resImage = cv2.putText(resImage, "X - " + str(poses[i][-1]),
                                           textPosition,
                                           cv2.FONT_HERSHEY_SIMPLEX,
                                           0.8,
                                           colors[i],
                                           2)
                else:
                    textPosition = (10, 20 + i * 25)
                    resImage = cv2.putText(resImage, str(poses[i][-1]),
                                           textPosition,
                                           cv2.FONT_HERSHEY_SIMPLEX,
                                           0.8,
                                           colors[i],
                                           2)

                # draw connection ellipses
                if i < poses.__len__() - 1:

                    if not math.isnan(poses[i][0]) and not math.isnan(poses[i][1]) and not \
                            math.isnan(poses[i+1][0]) and not math.isnan(poses[i+1][1]) \
                            and poses[i][0] != -1 and poses[i][1] != -1 \
                            and poses[i+1][0] != -1 and poses[i+1][1] != -1:
                        length = ((poses[i][1] - poses[i + 1][1]) ** 2 + (poses[i][0] - poses[i + 1][0]) ** 2) ** 0.5

                        angle = math.degrees(math.atan2(poses[i][1] - poses[i + 1][1], poses[i][0] - poses[i + 1][0]))

                        meanX = (poses[i][1] + poses[i + 1][1]) / 2
                        meanY = (poses[i][0] + poses[i + 1][0]) / 2

                        polygon = cv2.ellipse2Poly((int(meanY), int(meanX)), (int(length / 2), 2), int(angle), 0,
                                                   360, 1)

                        cv2.fillConvexPoly(resImage, polygon, colors[i])

        return resImage

    @staticmethod
    def DrawCircles(image, poses, colors=None, alpha=1.0):
        resImage = image
        if colors is None:
            colors = [[255, 255, 255] for i in range(0, poses.__len__())]

        for i in range(0, poses.__len__()):
            if not math.isnan(poses[i][0]) and not math.isnan(poses[i][1]):
                # draw pose circles
                position = (int(poses[i][0]), int(poses[i][1]))
                resImage = cv2.circle(resImage, position, 5, colors[i], thickness=-1)

        return resImage

    @staticmethod
    def DrawHumanPose(image, poses, colors, alpha=1.0):
        resImage = image
        for i in range(0, poses.__len__()):
            if not math.isnan(poses[i][0]) and not math.isnan(poses[i][1]):
                # draw pose circles
                position = (int(poses[i][0]), int(poses[i][1]))
                resImage = cv2.circle(resImage, position, 5, colors[i], thickness=-1)

                textPosition = (position[0] + 0, position[1] + 20)

            #draw limps
                # draw connection ellipses
            for root in config.humanPairMap:
                for index in root:
                    partA = poses[index]
                    partA = (int(partA[0]), int(partA[1]))

                    partB = poses[root[index]]
                    partB = (int(partB[0]), int(partB[1]))

                if partA[0] != -1 and partB[0] != -1:
                    resImage = cv2.line(resImage, partA, partB, colors[index], thickness=2)


        return resImage


    @staticmethod
    def PredictionToHeatmap(pred: np.array, normalize: bool=False):

        heatmaps = pred[0, :, :]

        for i in range(1, pred.shape[0]):
            heatmaps = np.maximum(heatmaps, pred[i, :, :])

        if normalize:
            heatmaps = Util.normalizeHeatmap(heatmaps)
        # cvHeatmaps = detection.boundingBox.CropImage(drawFrame)
        cvHeatmaps = Util.HeatmapToCV(heatmaps)

        return cvHeatmaps

    @staticmethod
    def HeatmapToCV(heatmap):
        heatmap = heatmap * 255
        heatmap = heatmap.astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return heatmap

    @staticmethod
    def NPToCVImage(image: np.array) -> np.array:
        image = image * 255
        image = image.astype(np.uint8)
        return image

    @staticmethod
    def CreateOverlayingImage(image, heatmap, weight: float = 0.5):

        if image.shape[0:2] != heatmap.shape[0:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        resImage = cv2.addWeighted(image, 1.0 - weight, heatmap, weight, 0)

        return resImage
