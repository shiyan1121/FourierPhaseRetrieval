from __future__ import annotations

import numpy as np
import math
from scipy.fft import fft2, fftshift
from typing import List


from detectron2.utils import load_from_numpy_file, numpy_to_tensor
from detectron2.projects.segmentation.data import ImageSample
from detectron2.projects.segmentation.transforms import Transform, TransformList



class FieldGenerationDataMapper:
    def __init__(self,
                 transforms: List[Transform]):

        self.transforms = TransformList(transforms)


    def _phase_norm(self, data):
        data[data<0] += 2*math.pi
        return (data) / (2*math.pi)

    def _abs_norm(self, data):
        maximum = np.max(data)
        minimum = np.min(data)
        return (data-minimum) / (maximum - minimum)
    
    def input_fft(self, data, crop_size):
        data_pad = np.pad(data,
                           (((crop_size[0] - data.shape[0]) // 2, (crop_size[0] - data.shape[0]) // 2),
                            ((crop_size[1] - data.shape[1]) // 2, (crop_size[1] - data.shape[1]) // 2)),
                           mode='constant', constant_values=0)

        input_fft = fftshift(fft2(fftshift(data_pad)))

        amp_input_fft = np.abs(input_fft)
        phase_input_fft = np.angle(input_fft)

        return amp_input_fft, phase_input_fft


    def __call__(self, data: dict) -> ImageSample:
        img_name = data['case_name']
        U_data = np.load(data['U_image_path'])

        output_measured = U_data['output_measured_rigid']
        input = U_data['input']

        amp_input_fft, phase_input_fft = self.input_fft(input, output_measured.shape)

        amp_output = self._abs_norm(np.abs(output_measured)).astype(np.float32)[None]
        amp_input_fft = self._abs_norm(amp_input_fft).astype(np.float32)[None]
        phase_input_fft = self._phase_norm(phase_input_fft).astype(np.float32)[None]

        image = amp_output
        label = np.concatenate([amp_input_fft, phase_input_fft])

        sample = ImageSample(
            img_name = img_name,
            image = image,
            label = label,
        )

        if len(self.transforms)>0:
            self.transforms(sample)

        return ImageSample(
            img_name=sample.img_name,
            image=numpy_to_tensor(sample.image),
            label=numpy_to_tensor(sample.label),
        )




