from PIL import Image, ImageOps
import torch
from torch import Tensor, nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class SquareImageTensor:
    """Fit an RGB image to a square and convert it to a normalized tensor."""

    def __init__(self, size: int, imagenet_normalize: bool = False) -> None:
        self.size = size
        self.imagenet_normalize = imagenet_normalize
        self.mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> Tensor:
        image = ImageOps.pad(
            image,
            (self.size, self.size),
            method=Image.Resampling.BILINEAR,
        )
        tensor = (
            torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            .view(self.size, self.size, 3)
            .permute(2, 0, 1)
            .float()
            .div_(255)
        )
        if self.imagenet_normalize:
            tensor = (tensor - self.mean) / self.std
        return tensor
