"""Generate Grad-CAM explanations for trained brain-tumor classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .model import create_model
from .preprocessing import PreprocessingConfig, preprocess_image, read_image


def _last_convolutional_layer(model: torch.nn.Module) -> torch.nn.Module:
    layers = [
        module
        for module in model.backbone.modules()
        if isinstance(module, torch.nn.Conv2d)
    ]
    if not layers:
        raise RuntimeError("Unable to find a convolutional layer for Grad-CAM.")
    return layers[-1]


def _model_input(image: np.ndarray, image_size: int) -> torch.Tensor:
    processed = preprocess_image(
        image, image_size=image_size, config=PreprocessingConfig()
    )
    tensor = torch.from_numpy(processed.transpose(2, 0, 1)).float().div(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


def explain_image(
    checkpoint: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    device: str | None = None,
) -> dict[str, str | float]:
    """Create a Grad-CAM overlay and return the predicted class and confidence."""
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_data = torch.load(checkpoint, map_location=selected_device)
    training_config = checkpoint_data.get("training_config", {})
    architecture = checkpoint_data["architecture"]
    image_size = int(checkpoint_data.get("image_size", 200))
    class_names = tuple(checkpoint_data.get("class_names", ("glioma", "meningioma", "no_tumor", "pituitary")))

    model = create_model(
        architecture=architecture,
        num_classes=len(class_names),
        dropout=float(training_config.get("dropout", 0.4)),
        pretrained=False,
    )
    model.load_state_dict(checkpoint_data["model_state"])
    model.to(selected_device).eval()

    original = read_image(image_path)
    input_tensor = _model_input(original, image_size).to(selected_device)
    target_layer = _last_convolutional_layer(model)
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        class_index = int(probabilities.argmax(dim=1).item())
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(class_index)],
        )[0]

    heatmap = cv2.resize(
        np.asarray(grayscale_cam, dtype=np.float32),
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    heatmap_uint8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 0.55, colored_heatmap, 0.45, 0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Unable to write Grad-CAM overlay: {output}")

    return {
        "class": class_names[class_index],
        "confidence": float(probabilities[0, class_index].item()),
        "output_path": str(output),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = explain_image(args.checkpoint, args.image, args.output, args.device)
    print(f"Prediction: {result['class']} ({result['confidence']:.4f})")
    print(f"Saved: {result['output_path']}")
