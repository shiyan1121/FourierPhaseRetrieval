import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from abc import ABC
from typing import Tuple, List, Literal, Generator
from torch import nn

from detectron2.projects.segmentation.transforms.split_combine import BaseSplitCombiner
from ..data.data_sample import ImageSample


class SplitCombiner(BaseSplitCombiner):
    def __init__(
            self,
            crop_size: Tuple[int, int],
            stride: Tuple[int, int],
            combine_method: Literal['max', 'avr', 'gw'] = 'max',
            device: Literal['cpu', 'cuda'] = 'cpu',
            sigma: float = 0.25):

        self.crop_size = np.asarray(crop_size)
        self.stride = np.asarray(stride)
        self.combine_method = combine_method
        self.device = device
        self.sigma = sigma

        self.reset()

    def reset(self):
        self.data_sample = None
        self.output_seg = None
        self.patch_cnt = None
        self.cur_patch_idx = 0

    def split(self, data_sample: ImageSample) -> Generator:
        self.data_sample = data_sample

        image = data_sample.image
        label = data_sample.label
        if self.device != 'cpu':
            image = image.to(self.device)
            label = label.to(self.device)
        _,h,w = image.shape
        [nh, nw] = [int(x) for x in (np.ceil((np.asarray([h, w])-self.crop_size)/self.stride + 1))]

        self.coord_list = []
        for i in range(nh):
            for j in range(nw):
                coord = self._get_patch_coord((i, j), (nh, nw), (h, w), self.crop_size, self.stride)
                patch = image[..., coord[0]:coord[1], coord[2]:coord[3]]
                label_patch = label[..., coord[0]:coord[1], coord[2]:coord[3]]
                self.coord_list.append(coord)
                yield type(data_sample)(image=patch, label = label_patch)

    def patch_to_output(self, output, patch_cnt, img_shape, patch, start, end):
        if output is None:
            output_channel = patch.shape[0]
            output = torch.full(
                (output_channel,) + img_shape,
                -1000 if self.combine_method == 'max' else 0,
                dtype=patch.dtype,
                device=patch.device)

            if self.combine_method in ['avr', 'gw']:
                patch_cnt = torch.zeros(
                    (1,) + img_shape,
                    dtype=torch.uint8 if self.combine_method == 'avr' else torch.float32,
                    device=patch.device
                )

        # resize
        # print(start, end)
        if np.all(patch.shape[-2:] != self.crop_size):
            patch = nn.functional.interpolate(patch[None], size=tuple(self.crop_size), mode='bicubic', align_corners=True)[0]
        # patch to output
        if self.combine_method == 'avr':
            output[:, start[0]:end[0], start[1]:end[1]] += patch
            patch_cnt[:, start[0]:end[0], start[1]:end[1]] += 1
        elif self.combine_method == 'max':
            output[:, start[0]:end[0], start[1]:end[1]] \
                = torch.maximum(patch, output[:, start[0]:end[0], start[1]:end[1]])
        elif self.combine_method == 'gw':
            gaussian_map = torch.from_numpy(self.get_gaussian(self.crop_size, sigma=self.sigma)).to(patch.device)
            output[:, start[0]:end[0], start[1]:end[1]] += patch * gaussian_map
            patch_cnt[:, start[0]:end[0], start[1]:end[1]] += gaussian_map

        return output, patch_cnt


    def combine(self, patch_sample: ImageSample):
        img_shape = self.data_sample.img_size

        coord = self.coord_list[self.cur_patch_idx]
        self.cur_patch_idx +=1

        start = np.array(coord[::2])
        end = np.array(coord[1::2])


        self.output_seg, self.patch_cnt = self.patch_to_output(
            self.output_seg, self.patch_cnt, img_shape, patch_sample.pred, start, end)


    def get_gaussian(self, patch_size, sigma = 0.25):
        tmp = np.zeros(patch_size)
        coord = [x//2 for x in patch_size]
        sigmas = [x*sigma for x in patch_size]
        tmp[tuple(coord)] = 1
        gaussian_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_map /= np.max(gaussian_map)
        return gaussian_map

    def norm(self, output, patch_cnt):
        patch_cnt == patch_cnt.clip_(min=1) if self.combine_method == 'avr' else patch_cnt
        if self.combine_method in ['avr', 'gw']:
            output /=patch_cnt
        return output


    def return_output(self) -> ImageSample:

        output_sample = self.data_sample
        output_sample.pred = self.norm(self.output_seg, self.patch_cnt)

        self.reset()
        return output_sample


