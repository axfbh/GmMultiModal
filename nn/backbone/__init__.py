from torchvision.models.resnet import resnet50
from torchvision.models.resnet import resnet101
from .ops import Backbone

__all__ = [
    "resnet50",
    "resnet101",
    "Backbone",
]
