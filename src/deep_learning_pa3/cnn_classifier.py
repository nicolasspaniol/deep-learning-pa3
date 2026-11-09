"""A small CNN baseline for the 21-class camera-trap classification task."""

from torch import Tensor, nn


class CNNClassifier(nn.Module):
    """Classify RGB images using a compact CNN trained from scratch.

    Inputs should be batches of images with shape ``(N, 3, H, W)``. Adaptive
    pooling allows different image sizes without changing the classifier head.
    The output contains one logit per class.
    """

    def __init__(self, num_classes: int = 21) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images)
        features = features.flatten(start_dim=1)
        return self.classifier(features)
