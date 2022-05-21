from kinematic_tracker.tracking.Tracker import Tracker
from typing import List
from ropose.scripts.pytorch.colropose_base import *
from ropose.net.pytorch.TrainingHyperParams import TrainingHyperParams
from guthoms_helpers.filesystem.FileHelper import FileHelper
from guthoms_helpers.filesystem.DirectoryHelper import DirectoryHelper
from guthoms_helpers.common_stuff.ProgressBar import ProgressBar
from ropose_dataset_tools.DataSetLoader import LoadDataSet

upsampleOutput = True
drawHeatmaps = False
printTestTimers = False
saveExamples = False
showResults = False

outputPath = "/mnt/datastuff/TestExamples/colropose_tracking/"
fineTunedModelPath = "/mnt/datastuff/FineTuning/TunedModels/"
DirectoryHelper.CreateIfNotExist(fineTunedModelPath)

originalModelPath = "/mnt/datastuff/Final_Ropose/best/Yolo.pt"
yolo.LoadPretrainedModel(originalModelPath)
yolo.SaveModelWeights("/mnt/datastuff/FineTuning/TunedModels/OriginalModel.pt")
yolo.LoadPretrainedModel("/mnt/datastuff/FineTuning/TunedModels/OriginalModel.pt")
yolo.ActivateFinetuning()

#roposeDatasets = loader.LoadDataSets(config.roposeEvalDataPath, None, False)

roposeDatasets = loader.LoadDataSetsForFinetuning(config.roposeFineTuneDatasets, includeNones=True)

simTH = 0.4
invalidTime = 100000

roposeTracker = Tracker(similarityThreshold=simTH, invalidationTimeMs=invalidTime)
#humanTracker = Tracker(similarityThreshold=0.9, invalidationTimeMs=1000)
trackingRoposeRecoveryCounter = 0
trackingHumanRecoveryCounter = 0

collectorAmount = 5
augmentations = 0
rawImages: List[np.array] = []
roposeTrainSets: List[YoloDetection] = []
humanTrainSets: List[YoloDetection] = []
hyperParams = TrainingHyperParams()

hyperParams.modelName = "Yolo"
hyperParams.seed = 1337
hyperParams.epochs = 1

hyperParams.batchSize = 5

hyperParams.optimizer = "SGD"
hyperParams.optimizer_lr = 1e-5
hyperParams.optimizer_momentum = 0.0


def RetrainYolo(images: List[np.array], roposeTrainSets: List[YoloDetection], humanTrainSets: List[YoloDetection],
                yoloNet: Yolo):

    yoloData = []
    yoloData.extend(roposeTrainSets)
    yoloData.extend(humanTrainSets)

    if yoloData.__len__() > 0:

        for i in range(0, yoloData.__len__()):
            yoloData[i] = (images[i], yoloData[i])

        yoloNet.FineTuneOnYoloDetection(yoloData, hyperParams,
                                        augmentation=augmentations > 0,
                                        augmentationAmount=augmentations,
                                        saveExamples=saveExamples)
        pass

for epoch in range(1, 11):
    counter = 0
    for datasetIndex in ProgressBar(range(0, roposeDatasets.__len__())):
        print("Index " + str(datasetIndex))
        dataset = roposeDatasets[datasetIndex]

        if dataset is None:
            # reset tracker for a new dataset
            roposeTracker = Tracker(similarityThreshold=simTH, invalidationTimeMs=invalidTime)
            continue

        counter+=1
        print("Counter: " + str(counter))
        testTimer = Timer()

        testTimer.Start("Loading")

        rawFrame = cv2.imread(dataset.rgbFrame.filePath)
        rawFrame = cv2.cvtColor(rawFrame, cv2.COLOR_BGR2RGB)
        testTimer.Stop(printTestTimers)

        testTimer.Start("PrepareRawInput")
        #x, padding, resizeFactor = Util.PrepareRawInput(rawFrame, config.yolo_InputSize, printTestTimers=False)

        elapsed = testTimer.Stop(printTestTimers)

        #yolo detection
        testTimer.Start("Yolo")
        roposeDetections, humanDetections, elapsed = DetectYolo(yolo, rawFrame)

        yoloFrame = copy.deepcopy(rawFrame)

        bbRopose = []
        if roposeDetections.__len__() > 1:
            test = True

        for detection in roposeDetections:
            bbRopose.append(detection.boundingBox)
            yoloFrame = detection.boundingBox.Draw(yoloFrame, description="robot")

        bbHuman = []
        for detection in humanDetections:
            bbHuman.append(detection.boundingBox)
            yoloFrame = detection.boundingBox.Draw(yoloFrame, description="human")

        if showResults:
            cv2.imshow('rawYolo', yoloFrame)

        drawDebugFrame = copy.deepcopy(rawFrame)

        roposeBeforeTrackingRecovery = bbRopose.__len__()
        notRecognizedRopose = roposeTracker.GetNotKnownInstances(bbRopose, predict=True, usePose=False,
                                                                 useHistoryCount=5)

        for newDet in notRecognizedRopose:
            newDet.Draw(drawDebugFrame, "Robot")
            newYolo = YoloDetection(newDet, config.yolo_RoposeClassNum, 1.0)
            roposeTrainSets.append([newYolo])
            roposeDetections.append(newYolo)
            rawImages.append(rawFrame)

        if showResults:
            cv2.imshow('TrackingHelp', drawDebugFrame)

        elapsed = testTimer.Stop(printTestTimers)
        drawFrame = copy.copy(rawFrame)

        testTimer.Start("RoPose")
        roPoseRet = Ropose(roPoseNet, rawFrame, roposeDetections, prinTimer=False, upsampleOutput=upsampleOutput)
        testTimer.Stop(printTestTimers)

        #show rawHeatmaps:
        testTimer.Start("Show Raw Heatmaps")
        if drawHeatmaps:
            for det in range(0, roposeDetections.__len__()):
                detection = roposeDetections[det]
                padding = roPoseRet[4][det]
                pred = roPoseRet[3][det].cpu().numpy()
                heatmaps = Util.PredictionToHeatmap(pred)

                #cvHeatmaps = detection.boundingBox.CropImage(drawFrame)
                cvHeatmaps = Util.HeatmapToCV(heatmaps)
                cvHeatmaps = Util.UnpadImage(cvHeatmaps, padding)
                drawFrame = Util.DrawOverlappingHeatmaps(drawFrame, cvHeatmaps,
                                                         detection.boundingBox)
                test = True

        testTimer.Stop(printTestTimers)

        testTimer.Start("Draw Examples")
        #draw ropose
        if roPoseRet is not None:
            supervisions = []
            for keypoints in roPoseRet[0]:
                roposeModel = PoseModelRopose_2D()
                roposeModel.UpdatePoses(keypoints)
                supervisions.append(roposeModel)

            trackingRoposeRecoveryCounter += supervisions.__len__() - roposeBeforeTrackingRecovery
            sims = roposeTracker.AddSupervision(supervisions)
            print("RobotSims: " + str(sims))


        if showResults:
            roposeTracker.DrawInstances(drawFrame)
            roposeTracker.DrawBoundingBoxes(drawFrame, "robot", color=[0.0, 0.0, 255])

        testTimer.Stop(printTestTimers)

        currentFPS = np.round(fpsTracker.FinishRun(), decimals=2)

        print("Tracker FPS: ", str(currentFPS))
        print("--END OF FRAME--")
        print("")

        counter += 1
        if showResults:
            cv2.putText(drawFrame, 'Sample: ' + str(counter) + ' - FPS: ' + str(currentFPS), (20, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 0, 0), lineType=1, thickness=3)

            cv2.putText(drawFrame, 'RobotInstances: ' + str(roposeTracker.trackedInstances.__len__()), (20, 80),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 0, 0), lineType=1, thickness=3)

        drawFrame = cv2.cvtColor(drawFrame, cv2.COLOR_RGB2BGR)

        if saveExamples:
            cv2.imwrite(os.path.join(outputPath, str(counter) + ".jpg"), drawFrame)
            cv2.imwrite(os.path.join(outputPath, str(counter) + "_yolo.jpg"), yoloFrame)

        if showResults:
            cv2.imshow('image_tracked', drawFrame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("RoPose Recovered:" + str(trackingRoposeRecoveryCounter))
        print("Humans Recovered:" + str(trackingHumanRecoveryCounter))

        if len(rawImages) >= collectorAmount:
            assert(len(rawImages) == len(roposeTrainSets))
            RetrainYolo(rawImages, roposeTrainSets, humanTrainSets, yolo)
            rawImages.clear()
            roposeTrainSets.clear()
            humanTrainSets.clear()

    #time.sleep(0.25)
    yolo.SaveModelWeights(os.path.join(fineTunedModelPath, "FineTunedEpoch_" + str(epoch) + ".pt"))
    trackingRoposeRecoveryCounter = 0
    trackingHumanRecoveryCounter = 0
    #reset tracker.. yeah i know i need a reset method
    roposeTracker = Tracker(similarityThreshold=simTH, invalidationTimeMs=invalidTime)
    inputProvider.ReInit()

import ropose.validation.validateFinetunedYoloRopose
