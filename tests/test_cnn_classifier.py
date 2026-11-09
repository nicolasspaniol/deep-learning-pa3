import torch

from deep_learning_pa3.cnn_classifier import CNNClassifier


def test_cnn_classifier_outputs_one_logit_per_class() -> None:
    model = CNNClassifier()
    logits = model(torch.randn(2, 3, 64, 80))

    assert logits.shape == (2, 21)


def test_cnn_classifier_handles_different_image_sizes() -> None:
    model = CNNClassifier()

    assert model(torch.randn(1, 3, 32, 48)).shape == (1, 21)
    assert model(torch.randn(1, 3, 72, 96)).shape == (1, 21)


def test_cnn_classifier_is_under_parameter_limit() -> None:
    model = CNNClassifier()

    assert sum(parameter.numel() for parameter in model.parameters()) <= 12_000_000
