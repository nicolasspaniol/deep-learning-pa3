import torch

from deep_learning_pa3.resnet_classifier import ResNetClassifier


def test_resnet_classifier_outputs_one_logit_per_class() -> None:
    model = ResNetClassifier(pretrained=False).eval()

    with torch.inference_mode():
        logits = model(torch.randn(1, 3, 64, 64))

    assert logits.shape == (1, 21)


def test_resnet_classifier_is_under_parameter_limit() -> None:
    model = ResNetClassifier(pretrained=False)

    assert sum(parameter.numel() for parameter in model.parameters()) <= 12_000_000
