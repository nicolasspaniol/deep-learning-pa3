import torch
from torch import Tensor, nn

from deep_learning_pa3.grad_cam import GradCAM


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 3)

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images)
        return self.classifier(self.pool(features).flatten(start_dim=1))


def test_grad_cam_returns_normalized_heatmap_and_logits() -> None:
    model = TinyClassifier().eval()
    image = torch.randn(1, 3, 16, 20)

    heatmap, logits = GradCAM(model, model.features[1]).generate(image)

    assert heatmap.shape == (16, 20)
    assert logits.shape == (3,)
    assert torch.isfinite(heatmap).all()
    assert heatmap.min() >= 0
    assert heatmap.max() <= 1


def test_grad_cam_rejects_non_single_image_batch() -> None:
    model = TinyClassifier().eval()

    try:
        GradCAM(model, model.features[1]).generate(torch.randn(2, 3, 16, 16))
    except ValueError as error:
        assert "single" in str(error) or "(1, C, H, W)" in str(error)
    else:
        raise AssertionError("Expected a ValueError for a multi-image batch")
