import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from deep_learning_pa3.evaluation import evaluate_model


class PredictionFromInput(nn.Module):
    def forward(self, images: Tensor) -> Tensor:
        return torch.nn.functional.one_hot(images[:, 0].long(), num_classes=3).float()


def test_evaluate_model_computes_macro_f1_and_confusion_matrix() -> None:
    images = torch.tensor([[0.0], [1.0], [1.0], [2.0]])
    labels = torch.tensor([0, 0, 1, 1])
    batches = DataLoader(TensorDataset(images, labels), batch_size=2)

    confusion, macro_f1 = evaluate_model(
        PredictionFromInput(), batches, torch.device("cpu"), num_classes=3
    )

    assert confusion.tolist() == [[1, 1, 0], [0, 1, 1], [0, 0, 0]]
    assert macro_f1 == pytest.approx(7 / 18)
