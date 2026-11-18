"""Grad-CAM heatmaps for convolutional image classifiers.

Grad-CAM weights each target-layer feature map by the spatially averaged
gradient of a selected class score, sums the weighted maps, and applies ReLU.
Based on Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (2017, arXiv:1610.02391).
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GradCAM:
    """Compute a normalized Grad-CAM heatmap for a single input image."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

    def generate(
        self, image: Tensor, target_class: int | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return ``(heatmap, logits)`` for one ``(1, C, H, W)`` image.

        The heatmap is resized to the input image dimensions and normalized to
        ``[0, 1]``. If ``target_class`` is omitted, explain the top prediction.
        Returned tensors are detached and on the CPU.
        """
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Grad-CAM expects an image batch with shape (1, C, H, W)")

        activations: Tensor | None = None
        gradients: Tensor | None = None

        def capture_activation(
            _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor
        ) -> None:
            nonlocal activations, gradients
            if not isinstance(output, Tensor) or output.ndim != 4:
                raise TypeError("Grad-CAM target layer must produce a 4D tensor")
            activations = output

            def capture_gradient(gradient: Tensor) -> Tensor:
                nonlocal gradients
                gradients = gradient
                return gradient

            output.register_hook(capture_gradient)

        hook = self.target_layer.register_forward_hook(capture_activation)
        self.model.zero_grad(set_to_none=True)
        try:
            with torch.enable_grad():
                logits = self.model(image.detach().requires_grad_(True))
                if not isinstance(logits, Tensor) or logits.ndim != 2:
                    raise TypeError("Grad-CAM model must return a (batch, classes) tensor")
                if activations is None:
                    raise RuntimeError("The target layer did not run during the model forward pass")

                class_index = (
                    int(logits[0].argmax().item())
                    if target_class is None
                    else target_class
                )
                if not 0 <= class_index < logits.shape[1]:
                    raise ValueError(f"target_class must be in [0, {logits.shape[1]})")

                logits[0, class_index].backward()
                if gradients is None:
                    raise RuntimeError("No gradients were captured from the target layer")

                weights = gradients.mean(dim=(2, 3), keepdim=True)
                heatmap = (weights * activations).sum(dim=1, keepdim=True).relu()
                heatmap = F.interpolate(
                    heatmap,
                    size=image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                heatmap = heatmap - heatmap.amin(dim=(2, 3), keepdim=True)
                heatmap = heatmap / heatmap.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)
                return heatmap[0, 0].detach().cpu(), logits[0].detach().cpu()
        finally:
            hook.remove()
            self.model.zero_grad(set_to_none=True)
