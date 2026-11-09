"""ImageNet-initialized ResNet-18 classifier for the 21 animal classes."""

from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNetClassifier(nn.Module):
    """Fine-tune a ResNet-18 by replacing its final layer with 21 logits.

    All backbone parameters remain trainable. By default, ImageNet-1K weights
    are used (downloaded by torchvision on first use if they are not cached).
    Set ``pretrained=False`` to initialize the network from scratch.
    """

    def __init__(self, num_classes: int = 21, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        return self.backbone(images)
