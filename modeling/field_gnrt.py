import numpy as np
from typing import Optional, Tuple, List, Union
import torch
from torch import nn
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import os
from detectron2.modeling.backbone import Backbone
from detectron2.utils.events import get_event_storage
from detectron2.projects.segmentation.data import ImageSample


class FieldGenerator(nn.Module):
    def __init__(self,
                 *,
                 backbone: Backbone,
                 head: nn.Module,
                 pixel_mean: float,
                 pixel_std: float,
                 resize_size: Union[None, int],
                 output_dir: ''
                 ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.resize_size = resize_size
        self.output_dir = output_dir


    @property
    def device(self):
        return self.pixel_mean.device

    def resize(self, image: torch.Tensor, size: tuple):
        return nn.functional.interpolate(image,size=size,mode='bicubic', align_corners=True)


    def preprocess_image(self, samples: List[ImageSample]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        images = [x.image.to('cuda')[None] for x in samples]
        images = F.normalize(torch.cat(images, dim=0), self.pixel_mean, self.pixel_std)
        images = self.resize(images, (self.resize_size,self.resize_size)) if self.resize_size is not None else images

        if self.training and samples[0].label is not None:
            targets = [x.label.to('cuda')[None] for x in samples]
            targets = torch.cat(targets, dim=0)
            targets = self.resize(targets, (self.resize_size,self.resize_size)) if self.resize_size is not None else targets
            if samples[0].condition is not None:
                conditions = torch.cat([x.condition.to('cuda')[None] for x in samples], dim=0)
            else:
                conditions = None
        else:
            targets = None
            conditions = None

        return images, targets, conditions

    def inference(self, images: torch.Tensor, targets: torch.Tensor, conditions: torch.Tensor=None):
        x = self.backbone(images)
        if conditions is not None:
            return self.head(x, targets, conditions)
        else:
            return self.head(x, targets)

    def forward(self, samples: List[ImageSample]) -> List[ImageSample]:
        images, targets, conditions = self.preprocess_image(samples)

        if self.training:
            storage = get_event_storage()
            self.iter = storage.iter

            logits, losses = self.inference(images, targets, conditions)
            self.prelosses = losses

            if self.iter % 500 == 0:
                self.save_img(samples[0].img_name, [x[0] for x in logits] if isinstance(logits, list) else [logits[0]], \
                              images[0], targets[0], losses, is_save=True)
            del logits

            return losses
        else:
            results, _ = self.inference(images, targets)

        # print('results.shape: {}'.format(results.shape))
        for result, sample in zip(results, samples):
            _, h, w = sample.label.shape
            # sample.pred = result
            sample.pred = self.resize(result[None], (h,w))[0] if self.resize_size is not None else result

        return samples

    def save_img(self, img_name, logits, image, target, losses, is_save=False):
        if not is_save:
            return

        num_logit = len(logits)
        assert len(logits[0].shape) == 3, f'logit.shape should be 3 dimensions, but got {len(logits[0].shape)}.'
        assert len(target.shape) == 3, f'target.shape should be 3 dimensions, but got {len(target.shape)}.'
        assert len(image.shape) == 3, f'image.shape should be 3 dimensions, but got {len(image.shape)}.'

        if num_logit>1:
            logits = [torch.permute(logit, (1, 2, 0)).detach().cpu().numpy().astype(np.float32) for logit in logits]
        else:
            logits = [lgt.unsqueeze(-1).detach().cpu().numpy().astype(np.float32) for lgt in logits[0]]
        image = [img.unsqueeze(-1).detach().cpu().numpy().astype(np.float32) for img in image]
        target = [tgt.unsqueeze(-1).detach().cpu().numpy().astype(np.float32) for tgt in target]

        loss_disp = {}
        for k, v in losses.items():
            if isinstance(v, list):
                loss_disp[k] = ', '.join([f'{vv:.4f}' for vv in v])
            else:
                loss_disp[k] = f'{v:.4f}'

        loss_disp = ', '.join([f'{k}: {v}' for k, v in loss_disp.items()])

        titles = [f'input {i}' for i in range(len(image))]+[f'label {i}' for i in range(len(target))]+[f'output {i}' for i in range(len(logits))]
        # titles.extend(['phase, label']+[f'phase, pred 0_{i+1}' for i in range(num_logit)])
        imgs = image + target + [x for x in logits]

        col = len(logits) + len(image) + len(target)
        fig, axes = plt.subplots(1, col, figsize=(col* 2.5, 3), layout='tight')
        for i, (img, subtitle) in enumerate(zip(imgs,titles)):
            # img = img[:, :, 1] if len(img.shape) == 3 and img.shape[-1]>1 and 'phase' in subtitle else img
            axes[i].imshow(img)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(subtitle)
        fig.suptitle(f'iter {self.iter}, {img_name}, {loss_disp}')

        os.makedirs(os.path.join(self.output_dir,'train'), exist_ok=True)
        plt.savefig(os.path.join(self.output_dir,'train',f'iter{self.iter}_{img_name}.jpg'))

