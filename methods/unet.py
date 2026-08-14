from __future__ import print_function, division

import segmentation_models_pytorch as smp

from methods.dl_common import DLSegmentationMethod, WrappedSegmentationModel


class UNetMethod(DLSegmentationMethod):
    def __init__(self, pretrained=True, use_augmentation=True):
        super().__init__(
            model=unet(pretrained=pretrained),
            model_name="unet",
            pretrained=pretrained,
            use_augmentation=use_augmentation,
        )


def unet(pretrained):
    # Use the same ResNet50 encoder family as the other deep learning baselines.
    if pretrained:
        network = smp.Unet("resnet50", classes=2)
    else:
        network = smp.Unet("resnet50", classes=2, encoder_weights=None)

    return WrappedSegmentationModel("unet", network)
