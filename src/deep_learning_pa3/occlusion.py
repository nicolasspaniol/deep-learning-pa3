"""Occlusion sensitivity maps for image classifiers."""

import torch
from torch import Tensor, nn


class Occlusion:
    """Estimate image-region importance by replacing patches with a baseline.

    Each heatmap value is based on how much masking an overlapping patch
    lowers the target class probability. A zero baseline corresponds to the
    ImageNet mean color when inputs use ImageNet normalization.
    """

    def __init__(
        self,
        model: nn.Module,
        window_size: int = 32,
        stride: int = 16,
        batch_size: int = 32,
        baseline: float = 0.0,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if stride < 1:
            raise ValueError("stride must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.model = model
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.baseline = baseline

    def generate(
        self, image: Tensor, target_class: int | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return ``(heatmap, logits)`` for one ``(1, C, H, W)`` image.

        The normalized heatmap is on the CPU, as are the detached original
        logits. With no target specified, the top predicted class is explained.
        """
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Occlusion expects an image batch with shape (1, C, H, W)")

        height, width = image.shape[-2:]
        window_height = min(self.window_size, height)
        window_width = min(self.window_size, width)
        y_positions = self._positions(height, window_height)
        x_positions = self._positions(width, window_width)
        windows = [(y, x) for y in y_positions for x in x_positions]

        with torch.inference_mode():
            logits = self.model(image)
            if not isinstance(logits, Tensor) or logits.ndim != 2:
                raise TypeError("Occlusion model must return a (batch, classes) tensor")

            class_index = (
                int(logits[0].argmax().item()) if target_class is None else target_class
            )
            if not 0 <= class_index < logits.shape[1]:
                raise ValueError(f"target_class must be in [0, {logits.shape[1]})")
            original_score = logits.softmax(dim=1)[0, class_index]

            heatmap = torch.zeros((height, width), device=image.device, dtype=image.dtype)
            counts = torch.zeros_like(heatmap)
            for start in range(0, len(windows), self.batch_size):
                batch_windows = windows[start : start + self.batch_size]
                occluded = image.expand(len(batch_windows), -1, -1, -1).clone()
                for index, (top, left) in enumerate(batch_windows):
                    occluded[
                        index,
                        :,
                        top : top + window_height,
                        left : left + window_width,
                    ] = self.baseline

                occluded_logits = self.model(occluded)
                if not isinstance(occluded_logits, Tensor) or occluded_logits.ndim != 2:
                    raise TypeError("Occlusion model must return a (batch, classes) tensor")
                score_drops = (
                    original_score - occluded_logits.softmax(dim=1)[:, class_index]
                ).clamp_min(0)

                for score_drop, (top, left) in zip(score_drops, batch_windows):
                    heatmap[
                        top : top + window_height,
                        left : left + window_width,
                    ] += score_drop
                    counts[
                        top : top + window_height,
                        left : left + window_width,
                    ] += 1

            heatmap = heatmap / counts.clamp_min(1)
            heatmap = heatmap - heatmap.min()
            heatmap = heatmap / heatmap.max().clamp_min(1e-8)
            return heatmap.cpu(), logits[0].cpu()

    def _positions(self, length: int, window: int) -> list[int]:
        positions = list(range(0, length - window + 1, self.stride))
        final_position = length - window
        if not positions or positions[-1] != final_position:
            positions.append(final_position)
        return positions
