import ssl
from functools import reduce
from typing import Tuple

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tsfm

ssl._create_default_https_context = ssl._create_unverified_context


def preproc_fn(h=224, w=224, rgb=True):
    """Get preprocessing function for imagenet.

    Args:
        h (int, optional): image height.
        w (int, optional): image width.
        rgb (bool, optional): if the input is a rgb image.
        input_channels (int, optional): input channels of the image.
    """
    # standard imagenet normalization taken from
    # https://github.com/pytorch/examples/blob/main/imagenet/main.py
    fns = []

    if rgb:
        fns = fns + [
            lambda x: x.float() / 255.0,
            tsfm.Resize([h, w]),
            tsfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    else:
        fns = fns + [tsfm.Resize([h, w])]

    return tsfm.Compose(fns)


def infer_model_out_shape(model, input_shape: Tuple) -> tuple:
    """Infer the output shape of model with given input shape.

    Returns:
        output shape
    """
    batch_shape = (1,) + input_shape
    with torch.no_grad():
        test_input = torch.rand(batch_shape)
        test_out = model(test_input)
    return tuple(list(test_out.size())[1:])


class ResnetEncoder(nn.Module):
    """ResNet variant encoder for image encoding."""

    def __init__(
        self,
        input_shape: tuple = (3, 224, 224),
        model: str = "resnet18",
        pretrained: bool = True,
        freeze: bool = True,
        rgb: bool = False,
    ):
        """Resnet encoder initialisation.

        Args:
            input_shape: input image shape
            model: resnet model name, choices=["resnet18", "resnet50", "resnet101"]
            pretrained: to use the imagenet pretrained model weight
            freeze: whether to freeze encoder
        """
        super().__init__()
        self.model = model
        self.pretrained = pretrained and (input_shape[0] == 3)
        self.freeze = freeze
        self.pre_proc = preproc_fn(*input_shape[-2:], rgb=rgb)

        if input_shape[0] != 3:
            self.channel_conv = nn.Conv2d(
                input_shape[0], 3, kernel_size=1, stride=1, padding=0
            )

        try:
            resnet = getattr(tv.models, model)(pretrained=self.pretrained)
        except AttributeError as e:
            raise NotImplementedError(f"No such network name {model}: {e}") from e

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        output_shape = infer_model_out_shape(self, input_shape)
        self.n_channel = reduce(lambda x, y: x * y, output_shape)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """NN forward steps.

        Args:
            rgb: input image tensor following pytorch channel order (C, H, W)

        Returns:
            encoded feature tensor
        """
        img = self.pre_proc(img)
        if img.size(1) != 3:
            img = self.channel_conv(img)

        if self.freeze:
            with torch.no_grad():
                feat = self.backbone(img)
        else:
            feat = self.backbone(img)
        return feat.squeeze(-1).squeeze(-1)
