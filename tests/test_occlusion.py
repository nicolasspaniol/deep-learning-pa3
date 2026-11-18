import torch
from torch import Tensor, nn

from deep_learning_pa3.occlusion import Occlusion


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


def test_occlusion_returns_normalized_heatmap_and_logits() -> None:
    model = TinyClassifier().eval()
    image = torch.randn(1, 3, 16, 20)

    heatmap, logits = Occlusion(
        model, window_size=8, stride=6, batch_size=4
    ).generate(image)

    assert heatmap.shape == (16, 20)
    assert logits.shape == (3,)
    assert torch.isfinite(heatmap).all()
    assert heatmap.min() >= 0
    assert heatmap.max() <= 1


def test_occlusion_rejects_invalid_inputs() -> None:
    model = TinyClassifier().eval()
    explainer = Occlusion(model, window_size=8)

    try:
        explainer.generate(torch.randn(2, 3, 16, 16))
    except ValueError as error:
        assert "(1, C, H, W)" in str(error)
    else:
        raise AssertionError("Expected a ValueError for a multi-image batch")

    try:
        explainer.generate(torch.randn(1, 3, 16, 16), target_class=3)
    except ValueError as error:
        assert "target_class" in str(error)
    else:
        raise AssertionError("Expected a ValueError for an invalid class index")
