import random
from typing import Literal, Sequence, Union, Dict, Any, List, overload
from itertools import chain

import numpy as np

import cv2
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2


class LongestMaxSize:
    def __init__(self,
                 max_size: Union[int, Sequence[int]] = 1024,
                 interpolation: int = cv2.INTER_LINEAR,
                 p: float = 1):
        T = [
            A.LongestMaxSize(max_size=max_size, interpolation=interpolation, p=p)
        ]

        self.fit_transform = A.Compose(T)

    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)


class Normalize:
    def __init__(self,
                 mean: None = (0.485, 0.456, 0.406),
                 std: None = (0.229, 0.224, 0.225),
                 max_pixel_value: Union[float, None] = 255.0,
                 normalization: Literal[
                     "standard", "image", "image_per_channel", "min_max", "min_max_per_channel"] = "standard",
                 p: float = 1.0):
        T = [
            A.Normalize(mean, std, max_pixel_value, normalization, p),
            ToTensorV2()
        ]
        self.fit_transform = A.Compose(T)

    def __call__(self, *args, **kwargs):
        return self.fit_transform(image=kwargs['image'])['image']
