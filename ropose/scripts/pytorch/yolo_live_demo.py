import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from ropose.net.pytorch.Yolo import Yolo
from ropose.net.pytorch.Util import Util, Timer
import ropose.pytorch_config as config
from guthoms_helpers.input_provider.ImageDirectory import ImageDirectory
from guthoms_helpers.common_stuff.FPSTracker import FPSTracker
from guthoms_helpers.common_helpers.ColorHelper import ColorHelper
import cv2
from ropose.net.pytorch.DatasetTrainingUtils import DatasetUtils


datasetUtil = DatasetUtils(onTheFlyBackgroundAugmentation=False, onTheFlyForegroundAugmentation=False)

if __name__ == "__main__":

    yolo = Yolo(netResolution=config.yolo_InputSize)
    yolo.LoadPretrainedModel(config.roPoseYoloWeights)
    #inputProvider = Camera(0, fps=30, useBuffer=False)
    dirPath = config.roposeEvalDataPath + "colropose_eval_005/depthcam1/rgb0"

    inputProvider = ImageDirectory(dirPath=dirPath)

    fpsTracker = FPSTracker()

    counter = 0

    window1 = cv2.namedWindow('image_tracked')
    window2 = cv2.namedWindow('image_raw')

    colors = ColorHelper.GetUniqueColors(config.yolo_Classes.__len__())

    while not inputProvider.finished():
        testTimer = Timer()

        testTimer.Start("Loading")
        rawFrame = inputProvider.GetData()

        rawFrame = cv2.resize(rawFrame, dsize=None, fx=0.25, fy=0.25)

        rawFrame = cv2.cvtColor(rawFrame, cv2.COLOR_BGR2RGB)

        detections, padding, resizeFactor = Util.PredictYolo(rawFrame, yoloNet=yolo, augment=False)

        drawFrame = rawFrame

        for detection in detections:
            if detection is not None:
                detection = Util.RestoreYoloBB(detection, resizeFactor, padding)
                drawFrame = detection.boundingBox.Draw(drawFrame, color=colors[detection.predictedClass],
                                                       description=Util.GetYoloClassName(detection.predictedClass))


        fps = fpsTracker.FinishRun()
        print("FPS: " + str(fps))
        drawFrame = cv2.cvtColor(drawFrame, cv2.COLOR_RGB2BGR)
        cv2.imshow('image_tracked', drawFrame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




