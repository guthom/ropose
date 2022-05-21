import torch
from torch import nn

class MultipleHeatmapLoss(nn.Module):
    def __init__(self, weight: float=1):
        super().__init__()
        self.weight = weight
        self.mseLoss = nn.MSELoss(reduction='mean')

    def HeamtapMSE(self, output, target) -> torch.Tensor:
        mse = self.mseLoss(output, target)
        return mse

    def forward(self, output, target):

        batches = output.size(0)
        joints = output.size(1)

        loss = torch.tensor(0.0).to(output.device)
        for i in range(0, batches):
            for j in range(0, joints):
                _mse = self.HeamtapMSE(output[i, j, :, :], target[i, j, :, :])
                loss += _mse

        loss = (loss / batches) * self.weight
        return loss
