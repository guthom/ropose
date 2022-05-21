# this class wraps the original model from the paper "Simple Baselines for Human Pose Estimation and Tracking"
# Have a look at the paper -> https://arxiv.org/abs/1804.06208
# Have a look at the original sourcecode -> https://github.com/microsoft/human-pose-estimation.pytorch
'''
@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}
'''

from ropose.thirdparty.humanpose.lib.models.pose_resnet import get_pose_net
from typing import List, Dict
import torch
import ropose.pytorch_config as config
from ropose.net.pytorch.NetBase import NetBase
import ropose.thirdparty.humanpose.lib.core.config as originalModelConfig

class HumanPoseNetOriginal(NetBase):

    def __init__(self, origWeightsPath: config.originalHumanPoseModelPath, netResolution: List = None):
        super().__init__(trainSet=None, testSet=None, validationSet=None, netResolution=netResolution,
                         modelName="Human_SimpleBaseLineOriginal")
        self.origWeightsPath = origWeightsPath

        self.LoadOriginalWeigths()


    def DefineModel(self):
        # override orrigin config for our purposes
        originalModelConfig.config.MODEL.NUM_JOINTS = 17
        originalModelConfig.POSE_RESNET.NUM_LAYERS = 152
        originalModelConfig.config.MODEL.IMAGE_SIZE = [config.inputRes[1], config.inputRes[0]]

        self.netModel = get_pose_net(cfg=originalModelConfig.config, is_train=False).to(self.device)
        pass

    def PreprocessInput(self, inp):
        return self.preprocessTransform(inp)

    def LoadOriginalWeigths(self):
        stateDict = torch.load(self.origWeightsPath)
        self.netModel.load_state_dict(stateDict)

    def Predict(self, input):
        _input = self.PreprocessInput(inp=input)

        ret = None
        with torch.no_grad():
            _input = _input.to(self.device)
            ret = self.netModel(_input)

        return ret