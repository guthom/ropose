from PIL import Image
import keras_config as config
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.cm as mplColormaps
import numpy as np
import sys
from scipy import misc
from ropose.net.keras.Util import Util as util

import cv2

class Display:
    @staticmethod
    def ShowImage(img):
        img = Image.fromarray(img)
        img.show()
        img.close()

    @staticmethod
    def ShowArrayScy(img):
        misc.imshow(img)

    @staticmethod
    def ShowDataset(dataset, grayScale=False):
        rawImage, rawGroundTruth, rawGroundTruth_punish= util.LoadXYToArray(dataset)

        picture = Display.drawBoundingBox(rawImage[0], dataset, grayScale=grayScale)
        picture = Display.drawMarker(picture, dataset, grayScale=grayScale)
        Display.ShowImage(picture)

        heatmap = util.CreateHeatmapFromGroundTruth(dataset)
        Display.ShowArrayScy(heatmap)
        return

    @staticmethod
    def ShowDetectedPoses(image, dataset, detectedPoses, drawGT=False):
        rawImage = image

        if drawGT:
            rawImage = Display.drawMarker(rawImage, dataset['joint_pos2D'], color=(0, 255, 0))

        Display.drawMarker(rawImage, detectedPoses, asPose=True, color=(255, 0, 0))

        Display.ShowImage(image)

    @staticmethod
    def drawMarker(picture, coords, grayScale=False, asPose=False, color=(0, 0, 255)):
        for i in range(0, coords.__len__()):
            if asPose:
                if not coords[i].x == -1 and not coords[i].y == -1:
                    if not grayScale:
                        cv2.circle(picture, (int(coords[i].x), int(coords[i].y)), 3, color, -1)
                    else:
                        cv2.circle(picture, (int(coords[i].x), int(coords[i].y)), 3, 255, -1)
            else:
                if not coords[str(i)][0] == -1 and not coords[str(i)][1] == -1:
                    if not grayScale:
                        cv2.circle(picture, (int(coords[str(i)][0]), int(coords[str(i)][1])), 3, color, -1)
                    else:
                        cv2.circle(picture, (int(coords[str(i)][0]), int(coords[str(i)][1])), 3, 255, -1)

        return picture

    @staticmethod
    def drawBoundingBox(picture, poses, grayScale=False, color=(0, 0, 255)):
        if not grayScale:
            cv2.rectangle(picture, (int(poses[0][0]), int(poses[0][1])),
                          (int(poses[1][0]), int(poses[1][1])), color, thickness=3)
        else:
            cv2.rectangle(picture, (int(poses[0][0]), int(poses[0][1])),
                          (int(poses[1][0]), int(poses[1][1])), 255, thickness=3)
        return picture

    @staticmethod
    def ShowBoundingBox(img, poses, grayScale=False):
        picture = Display.drawBoundingBox(img, poses, grayScale=grayScale)
        Display.ShowImage(picture)

    @staticmethod
    def ShowPoses(img, poses, grayScale=False):
        picture = Display.drawMarker(img, poses, grayScale=grayScale)
        Display.ShowImage(picture)

    @staticmethod
    def startProgress(title):
        global progress_x
        sys.stdout.write(title + ": [" + "-" * 40 + "]" + chr(8) * 41)
        sys.stdout.flush()
        progress_x = 0

    @staticmethod
    def progress(x):
        global progress_x
        x = int(x * 40 // 100)
        sys.stdout.write("#" * (x - progress_x))
        sys.stdout.flush()
        progress_x = x


    @staticmethod
    def PlotJointInputMaps(image, jointInput):

        fig = plt.figure(figsize=(10, 10))
        subPlot = fig.add_subplot(1, 1, 1)

        jointInput = np.transpose(jointInput, (2, 0, 1))
        arr = np.zeros((352, 352))
        for jointMap in jointInput:
            arr += jointMap

        #plt.imshow(image)
        plt.gca().invert_yaxis()
        max = np.max(arr)
        min = np.min(arr)
        # plt.contourf(arr, cmap=cmap, vmin=config.heatmapMinValue, vmax=np.max(arr))
        plt.imshow(arr, alpha=0.5)


        plt.show()
        return

    @staticmethod
    def PlotPredictedSet(image, gt, heatmaps, split=False, overLap=False, upsampling=True, appendGT=True):
        if upsampling:
            upsampledHeatmap = (util.UpsampleHeatmapsThreaded(heatmaps, image.shape[0:2]))
            upsampledGT = (util.UpsampleHeatmapsThreaded(gt, image.shape[0:2], normalize=True))

            heatmaps = upsampledHeatmap
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
                for arr in heatmaps:
                    plt.gca().invert_yaxis()
                    max = np.max(arr)
                    min = np.min(arr)
                    # plt.contourf(arr, cmap=cmap, vmin=config.heatmapMinValue, vmax=np.max(arr))
                    plt.imshow(arr, cmap=cmap, vmin=min, vmax=max, alpha=0.5)

            else:
                subPlot = fig.add_subplot(4, 3, 1)
                plt.gca().invert_yaxis()
                plt.imshow(image)
                # create GT image
                if appendGT:
                    subPlot = fig.add_subplot(4, 3, 3)
                    subPlot.set_title("Ground truth heatmap")
                    together = util.OverlayHeatmaps(gt[0:-1])
                    together = util.normalizeHeatmap(together)

                    if upsampling:
                        plt.imshow(image, alpha=0.7)
                        plt.pcolormesh(together, cmap=cmap, vmin=np.min(together), vmax=np.max(together), alpha=0.2)
                    else:

                        plt.gca().invert_yaxis()
                        plt.pcolormesh(together, cmap=cmap, vmin=np.min(together), vmax=np.max(together))

                    #plot Background
                    subPlot = fig.add_subplot(4, 3, 2)
                    subPlot.set_title("Background heatmap")
                    heatmap = heatmaps[-1]

                    if upsampling:
                        plt.imshow(image, alpha=0.7)
                        plt.pcolormesh(heatmap, cmap=cmap, vmin=np.min(heatmap), vmax=np.max(heatmap), alpha=0.2)
                    else:
                        plt.pcolormesh(heatmap, cmap=cmap, vmin=np.min(heatmap), vmax=np.max(heatmap))


                for i in range(heatmaps.__len__() - 1):
                    subPlot = fig.add_subplot(4, 3, i+4)
                    maxVal = np.max(heatmaps[i])
                    subPlot.set_title(str(i) + " MaxVal: " + str(maxVal))

                    if upsampling:
                        plt.imshow(image, alpha=0.7)
                        plt.pcolormesh(heatmaps[i], cmap=cmap, vmin=np.min(heatmaps[i]), vmax=maxVal, alpha=0.2)
                    else:
                        plt.pcolormesh(heatmaps[i], cmap=cmap, vmin=np.min(heatmaps[i]), vmax=maxVal)

        else:
            fig = plt.figure(figsize=(10, 10))
            plt.gca().invert_yaxis()
            # plot raw image first
            subPlot = fig.add_subplot(1, 2, 1)
            plt.imshow(image)
            subPlot = fig.add_subplot(1, 2, 2)
            plt.imshow(heatmaps)

        plt.show()

        return

    @staticmethod
    def PlotSingleData(data):
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(data)

    @staticmethod
    def ShowSimpleProgress(progress, stats=None, batchCount=None, batchSize=16, elapsedTime=None):
        if batchCount is not None:
            batchString = str(batchCount[0]*batchSize+batchSize) + "/" + str(batchCount[1]*batchSize)
        else:
            batchString = ""

        if stats is not None:
            statsString = " " + str(stats)
        else:
            statsString = ""

        if elapsedTime is not None:
            seconds = elapsedTime.total_seconds()
            if batchCount[0] != 0.0:
                etaS = seconds * (batchCount[1] - batchCount[0])
                etaMin = etaS / 60
                etaH = etaMin / 60
                etaTime = datetime.now() + timedelta(seconds=etaS)

                #round numbers
                etaMin = round(etaMin, 2)
                etaH = round(etaH, 2)
            else:
                etaMin = None
                etaH = None
                etaTime = datetime.now()

            timeString = " - elapsed: " + str(seconds) + "s. ETA: " + str(etaMin, ) + \
                         "min. / " + str(etaH) + " h. - " + str(etaTime)

        else:
            timeString = ""

        if stats is not None:
            print(batchString + "[" + str(progress) + "%] " + statsString + timeString)

        return

    @staticmethod
    def DumpDataset(path, dataset, split=False, overLap=False, upsampling=True, grayScale=False):

        rawImage, rawGroundTruth, rawGroundTruth_punish = util.LoadXYToArray(dataset, config.preparedDataPath, grayScale=grayScale)

        image = rawImage[0]
        #image = Display.drawBoundingBox(rawImage[0], dataset['bBox'], grayScale=grayScale)
        #image = Display.drawMarker(image, dataset['joint_pos2D'], grayScale=grayScale)

        heatmaps = util.CreateHeatmapFromGroundTruth(dataset, split)

        if upsampling:
            upsampledHeatmap = []
            for heatmap in heatmaps:
                upsampledHeatmap.append(util.BiliniarUpsampling(heatmap, image.shape[0:2]))
            array = upsampledHeatmap

        if split:
            fig = plt.figure(figsize=(10, 10))
            # plot raw image first
            if overLap:
                subPlot = fig.add_subplot(1, 1, 1)
            else:
                subPlot = fig.add_subplot(4, 3, 1)

            cmap = mplColormaps.get_cmap('rainbow')
            cmap._init()
            alphas = np.linspace(0, 1.0, cmap.N + 3)
            cmap._lut[:, -1] = alphas

            if overLap:
                subPlot = fig.add_subplot(111)

                plt.imshow(image)
                for arr in array:
                    plt.contourf(arr, cmap=cmap, vmin=np.min(arr), vmax=np.max(arr))
                pass
            else:
                for i in range(array.__len__()):
                    subPlot = fig.add_subplot(4, 3, i+4)
                    maxVal = np.max(array[i])
                    subPlot.set_title(str(i) + " MaxVal: " + str(maxVal))

                    plt.pcolor(array[i], cmap=cmap, vmin=np.min(array[i]), vmax=maxVal, alpha=0.6)
        else:
            fig = plt.figure(figsize=(10, 10))
            # plot raw image first
            subPlot = fig.add_subplot(1, 2, 1)
            plt.imshow(image)
            subPlot = fig.add_subplot(1, 2, 2)
            plt.imshow(array)

        plt.savefig(path + "/" + dataset["filename"])

        plt.close()
        return

    @staticmethod
    def PlotDataset(dataset, index=0, split=False, overLap=False, upsampling=True):

        image = dataset[0][index]
        heatmaps = dataset[1][index]
        if upsampling:
            heatmaps = util.UpsampleHeatmapsThreaded(heatmaps)

        rawGroundTruth_punish = dataset[2][index]

        if split:
            fig = plt.figure(figsize=(10, 10))
            # plot raw image first
            if overLap:
                subPlot = fig.add_subplot(1, 1, 1)
            else:
                subPlot = fig.add_subplot(4, 3, 1)

            cmap = mplColormaps.get_cmap('rainbow')
            cmap._init()
            alphas = np.linspace(0, 1.0, cmap.N + 3)
            cmap._lut[:, -1] = alphas

            if overLap:
                subPlot = fig.add_subplot(111)

                plt.imshow(image)
                for arr in heatmaps:
                    plt.contourf(arr, cmap=cmap, vmin=np.min(arr), vmax=np.max(arr))
                pass
            else:

                subPlot = fig.add_subplot(4, 3, 1)
                subPlot.imshow(image)

                for i in range(heatmaps.__len__()):
                    subPlot = fig.add_subplot(4, 3, i + 4)
                    subPlot.imshow(image)
                    maxVal = np.max(heatmaps[i])
                    subPlot.set_title(str(i) + " MaxVal: " + str(maxVal))

                    plt.pcolor(heatmaps[i], cmap=cmap, vmin=np.min(heatmaps[i]), vmax=maxVal, alpha=0.6)
        else:
            fig = plt.figure(figsize=(10, 10))
            # plot raw image first
            subPlot = fig.add_subplot(1, 2, 1)
            plt.imshow(image)
            subPlot = fig.add_subplot(1, 2, 2)
            plt.imshow(heatmaps)

        plt.show()
        return

    @staticmethod
    def endProgress():
        sys.stdout.write("#" * (40 - progress_x) + "]\n")
        sys.stdout.flush()