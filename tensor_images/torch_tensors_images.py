"""
PyTorch — Tensors from Images (Starter)

Tasks:
1) Download/load CIFAR-10 using torchvision.datasets
2) Take ONE image (a PIL Image)
3) Convert it to a torch.Tensor of shape (3, H, W)
4) Display the tensor as an image to verify it looks correct
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


def get_dataset(root: Path) -> datasets.CIFAR10:
    """
    Returns the CIFAR-10 training set.

    The first time you run this, it will download the dataset to `root`.
    """
    root.mkdir(parents=True, exist_ok=True)
    return datasets.CIFAR10(root=str(root), train=True, download=True, transform=None)


def pil_to_tensor(pil_img) -> torch.Tensor:
    """
    Convert a PIL image (RGB) to a torch.Tensor with shape (3, H, W).
    """
    to_tensor = transforms.ToTensor()  # float32 in [0, 1], shape (C, H, W)
    return to_tensor(pil_img)


def show_tensor_image(img_tensor: torch.Tensor, title: str, save_path: Path | None = None) -> None:
    """
    Requirements:
    - img_tensor is a torch.Tensor with shape (3, H, W) and values in [0, 1]
    - matplotlib expects (H, W, C), so you must reorder dimensions before plotting

    Hints:
    - Use `img_tensor.permute(1, 2, 0)` to go from (C,H,W) -> (H,W,C)
    - Convert to numpy with `.detach().cpu().numpy()`
    """
    plt.imshow(img_tensor.permute(1, 2, 0).detach().cpu().numpy())
    plt.title(title)
    plt.show()


def image_tensor_to_vector(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Hints:
    - `img_tensor.flatten()` returns a 1D tensor
    - Keep the order consistent so that reshaping back recovers the original
    """
    return img_tensor.flatten()

def main() -> None:
    data_root = Path(__file__).resolve().parent / "data"
    dataset = get_dataset(data_root)

    index = 0
    pil_img, label = dataset[index]

    img_tensor = pil_to_tensor(pil_img)

    # Basic checks (do not remove)
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.ndim == 3, f"Expected a 3D tensor (C,H,W). Got shape={tuple(img_tensor.shape)}"
    assert img_tensor.shape[0] == 3, f"Expected 3 channels (RGB). Got shape={tuple(img_tensor.shape)}"

    class_name = dataset.classes[label] if hasattr(dataset, "classes") else str(label)
    print(f"{label}: {class_name}")
    print(f"Min: {img_tensor.min()}\tMax: {img_tensor.max()}")

    vec = image_tensor_to_vector(img_tensor)
    print(vec.shape)

    reshaped = vec.reshape((3, 32, 32))

    # Checks that reshaped vector is correct datatype
    assert isinstance(reshaped, torch.Tensor)
    assert reshaped.ndim == 3, f"Expected a 3D tensor (C,H,W). Got shape={tuple(reshaped.shape)}"
    assert reshaped.shape[0] == 3, f"Expected 3 channels (RGB). Got shape={tuple(reshaped.shape)}"

    show_tensor_image(img_tensor, title=f"CIFAR-10 — {class_name}")


if __name__ == "__main__":
    main()
