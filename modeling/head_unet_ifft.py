import logging
import torch
import torch.nn as nn
import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union, Callable, List

logger = logging.getLogger(__name__)

def _phase_norm(data):
    data[data < 0] += 2 * math.pi
    # data = torch.fmod(data, 2 * math.pi)
    return (data) / (2 * math.pi)

def WrapMSE(preds, labels):
    delta_phi = torch.remainder(preds - labels + 0.5, 1) - 0.5
    return torch.mean(delta_phi ** 2)

def MSE(preds, labels):
    loss = torch.nn.MSELoss()
    return loss(preds, labels)

def L1(preds, labels):
    loss = torch.nn.L1Loss()
    return loss(preds, labels)

class FieldGnrtUNetHead(nn.Module, ABC):
    def __init__(
        self,
        in_channel: int = 64,
        out_channel: int = 1,
        is_drop: bool = False,
        losses_ratio: dict = {'OutputAmpMSEloss': 1, 'OutputPhaseWrapMSEloss': 1}
    ):
        super().__init__()

        self.is_drop = is_drop
        self.losses_ratio = losses_ratio

        # sigmoid
        self.out_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel,kernel_size=1,stride=1),
            torch.nn.Sigmoid())

        if self.is_drop:
            self.drop_out = nn.Dropout(p = 0.5)

    def resize(self, image: torch.Tensor, size: tuple):
        return nn.functional.interpolate(image,size=size,mode='bicubic', align_corners=True)

    def forward(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None, conditions: Optional[torch.Tensor] = None):

        # drop layer
        if self.is_drop:
            features = self.drop_out(features)

        # out layer
        logits = self.out_layer(features).to(torch.float32)


        if self.training:
            return logits, self.losses(logits, targets, conditions)
        else:
            return logits, {}

    def compute_loss(self, preds, targets, loss_type='MSE'):
        if loss_type == 'MSE':
            loss = MSE(preds, targets)
        elif loss_type == 'L1':
            loss = L1(preds, targets)
        elif loss_type == 'WrapMSE':
            loss = WrapMSE(preds, targets)
        return loss

    def losses(self, logits: Union[List[torch.Tensor], torch.Tensor], targets: torch.Tensor, conditions: torch.Tensor):
        """
        define your losses
        """

        losses = {}
        losses['total_loss'] = 0
        for (k, pred, target) in zip(self.losses_ratio.keys(),
                        [logits[:, 0:1, :, :], logits[:, 1:2, :, :]],
                        [targets[:, 0:1, :, :], targets[:, 1:2, :, :]]):

            losses[k] = self.compute_loss(pred, target, loss_type=k.split('_')[-1][:-4])
            losses['total_loss'] += losses[k] * self.losses_ratio[k]

        assert type(losses) is dict, 'Losses should be a dict.'

        return losses

    def input_ifft(self, amp, phase):
        dtype = amp.dtype
        if dtype==torch.float16:
            amp = amp.to(torch.float32)
            phase = phase.to(torch.float32)

        field = amp * torch.exp(1j * phase)
        field_ifftshift = torch.fft.ifftshift(field)
        field_ifft = torch.fft.ifft2(field_ifftshift)

        if dtype==torch.float16:
            field_ifft = torch.real(field_ifft).to(dtype) + 1j*torch.imag(field_ifft).to(dtype)

        return torch.fft.ifftshift(field_ifft)


