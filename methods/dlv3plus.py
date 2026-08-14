import segmentation_models_pytorch as smp
import torch.nn as nn
from methods.dl_common import DLSegmentationMethod, WrappedSegmentationModel


class DLv3Plus(DLSegmentationMethod):
    def __init__(self, pretrained, use_augmentation=True):
        super().__init__(
            model=deeplabv3plus(pretrained=pretrained),
            model_name="deeplabv3plus",
            pretrained=pretrained,
            use_augmentation=use_augmentation,
        )


def deeplabv3plus(pretrained):
    # DeepLabV3+ acts as the stronger architecture-level comparison baseline.
    if pretrained:
        network = smp.DeepLabV3Plus('resnet50', classes=2)
    else:
        network = smp.DeepLabV3Plus('resnet50', classes=2, encoder_weights=None)

    return WrappedSegmentationModel("deeplabv3plus", network)
