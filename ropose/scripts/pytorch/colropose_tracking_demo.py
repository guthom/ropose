from kinematic_tracker.tracking.Tracker import Tracker
from guthoms_helpers.filesystem.DirectoryHelper import DirectoryHelper
from ropose.scripts.pytorch.colropose_base import *

upsampleOutput = True
upsamplingOriginalSize = True
drawHeatmaps = False
printTestTimers = False
saveExamples = False
saveResultVideo = True
showLive = True

outputPath = config.outputDir + "/colropose_tracking/"
DirectoryHelper.CreateIfNotExist(outputPath)
roposeTracker = Tracker(similarityThreshold=0.4, invalidationTimeMs=1000)

trackingRoposeRecoveryCounter = 0
trackingHumanRecoveryCounter = 0

withTracker = [0, 0]
withoutTracker = [0, 0]

videoWriter = None
if saveResultVideo:
    videoWriter = cv2.VideoWriter(outputPath + 'roposeResult.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                  fps=30, frameSize=(1280, 720))

if showLive:
    display = cv2.namedWindow('image_raw_estimated', cv2.WINDOW_NORMAL)
    display2 = cv2.namedWindow('rawYolo', cv2.WINDOW_NORMAL)
    display3 = cv2.namedWindow('TrackingHelp', cv2.WINDOW_NORMAL)
    display4 = cv2.namedWindow('image_tracked', cv2.WINDOW_NORMAL)

while not inputProvider.finished():
    testTimer = Timer()

    testTimer.Start("Loading")
    rawFrameBGR = inputProvider.GetData()
    rawFrame = cv2.cvtColor(rawFrameBGR, cv2.COLOR_BGR2RGB)

    if rawFrame is None:
        continue

    bgrInput = rawFrame
    testTimer.Stop(printTestTimers)

    #yolo detection
    testTimer.Start("Yolo")
    detections, padding, resizeFactor = Util.PredictYolo(rawFrame, yoloNet=yolo, augment=False)

    humanDetections, roposeDetections = Util.FilterYolo(detections)

    yoloFrame = copy.deepcopy(rawFrame)
    poseFrame = copy.deepcopy(rawFrameBGR)
    poseFrame = Util.ToFloat64Image(poseFrame)

    bbRopose = []
    if roposeDetections.__len__() > 1:
        test = True

    for detection in roposeDetections:
        detection = Util.RestoreYoloBB(detection, resizeFactor, padding)
        bbRopose.append(detection.boundingBox)
        yoloFrame = detection.boundingBox.Draw(yoloFrame, description="robot")

    bbHuman = []
    for detection in humanDetections:
        detection = Util.RestoreYoloBB(detection, resizeFactor, padding)
        bbHuman.append(detection.boundingBox)
        yoloFrame = detection.boundingBox.Draw(yoloFrame, description="human")

    if showLive:
        cv2.imshow('rawYolo', yoloFrame)

    drawDebugFrame = copy.deepcopy(rawFrame)

    roposeBeforeTrackingRecovery = bbRopose.__len__()
    withoutTracker[0] += bbRopose.__len__()
    notRecognizedRopose = roposeTracker.GetNotKnownInstances(bbRopose)
    withTracker[0] += bbRopose.__len__() + notRecognizedRopose.__len__()
    for newDet in notRecognizedRopose:
        newDet.Draw(drawDebugFrame, "Robot")
        newYolo = YoloDetection(newDet, config.yolo_RoposeClassNum, 1.0)
        roposeDetections.append(newYolo)

    humanBeforeTrackingRecovery = bbHuman.__len__()
    withoutTracker[1] += bbHuman.__len__()

    if showLive:
        cv2.imshow('TrackingHelp', drawDebugFrame)

    elapsed = testTimer.Stop(printTestTimers)
    drawFrame = copy.copy(rawFrame)

    testTimer.Start("RoPose")
    roPoseRet = Ropose(roPoseNet, rawFrame, roposeDetections, prinTimer=False, upsampleOutput=upsampleOutput,
                       upsamplingOriginalSize=upsamplingOriginalSize)
    testTimer.Stop(printTestTimers)

    testTimer.Start("HumanPose")
    #print("Raw detected Humans: ", str(humanDetections.__len__()))
    if humanDetections.__len__() > 0:
        test = True

    humanRet = HumanPose(humanNet, poseFrame, humanDetections, prinTimer=False, upsampleOutput=upsampleOutput)

    testTimer.Stop(printTestTimers)

    roposeTrackingPredictions = roposeTracker.Predict2DBB(5)
    roposeTrackingPredictionsPoses = roposeTracker.PredictChain(5)
    for trackingBB in roposeTrackingPredictions:
        #trackingBB.Draw(drawFrame, "tracked", [0, 255, 0])
        print(str(counter) + " " + str(trackingBB.toList()))
    for poseModel in roposeTrackingPredictionsPoses:
        #poseModel.GetBoundingBox().Draw(drawFrame, "tracked_pose", [0, 255, 255])
        pass

    for predictedBB in bbRopose:
        #predictedBB.Draw(drawFramedrawFrame, "predicted", [255, 0, 0])
        pass

    #show rawHeatmaps:
    testTimer.Start("Show Raw Heatmaps")
    if drawHeatmaps:
        for det in range(0, roposeDetections.__len__()):
            detection = roposeDetections[det]
            padding = roPoseRet[4][det]
            pred = roPoseRet[3][det].cpu().numpy()
            heatmaps = Util.PredictionToHeatmap(pred[0:7, :, :])

            #cvHeatmaps = detection.boundingBox.CropImage(drawFrame)
            cvHeatmaps = Util.HeatmapToCV(heatmaps)
            cvHeatmaps = Util.UnpadImage(cvHeatmaps, padding)
            if cvHeatmaps.shape[0] > 0 and cvHeatmaps.shape[1] > 0:
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
        #print("RobotSims: " + str(sims))

    #draw humans
    if humanRet is not None:
        supervisions = []
        for keypoints in humanRet[0]:
            humanModel = PoseModelHuman17_2D()
            humanModel.UpdatePoses(keypoints)
            supervisions.append(humanModel)
            humanModel.Draw(drawFrame)

        trackingHumanRecoveryCounter += supervisions.__len__() - humanBeforeTrackingRecovery

    # invalidate tracking instances
    #roposeTracker.CheckInvalidation()
    #humanTracker.CheckInvalidation()

    roposeTracker.DrawInstances(drawFrame)
    #roposeTracker.DrawBoundingBoxes(drawFrame, "robot", color=[0.0, 0.0, 255])

    testTimer.Stop(printTestTimers)

    currentFPS = np.round(fpsTracker.FinishRun(), decimals=2)

    #print("Tracker FPS: ", str(currentFPS))
    #print("--END OF FRAME--")
    #print("")

    counter += 1
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

    if showLive:
        cv2.imshow('image_tracked', drawFrame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if saveResultVideo:
        videoWriter.write(drawFrame)


    #print("RoPose Recovered:" + str(trackingRoposeRecoveryCounter))
    #print("Humans Recovered:" + str(trackingHumanRecoveryCounter))

    #time.sleep(0.25)

if saveResultVideo:
    videoWriter.release()

print("Before Tracking: " + str(withoutTracker))
print("With Tracking: " + str(withTracker))
