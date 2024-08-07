import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple, Union

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image

import os
import subprocess

_GEN_NAME = "pggan_celebahq1024"
_DEV = "cpu"

_MODEL_ZOO = {
    # PGGAN official.
    "pggan_celebahq1024": dict(
        # gan_type="pggan",
        resolution=1024,
        url="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EW_3jQ6E7xlKvCSHYrbmkQQBAB8tgIv5W5evdT6-GuXiWw?e=gRifVa&download=1",
    ),
}

from pggan_generator import PGGANGenerator

def load_generator(model_name: str, device: Any, checkpoints_dir: str) -> Any:
    """Loads pre-trained generator."""
    if model_name not in _MODEL_ZOO:
        raise KeyError(f"Unknown model name `{model_name}`!")

    model_config = _MODEL_ZOO[model_name].copy()
    url = model_config.pop("url")  # URL to download model if needed.

    # Build generator.
    print(f"Building generator for model `{model_name}` ...")
    # generator = build_generator(**model_config)
    generator = PGGANGenerator(**model_config)
    print("Finish building generator.")

    # Load pre-trained weights.
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoints_dir, model_name + ".pth")
    print(f"Loading checkpoint from `{checkpoint_path}` ...")
    if not os.path.exists(checkpoint_path):
        print(f"  Downloading checkpoint from `{url}` ...")
        subprocess.call(["wget", "--quiet", "-O", checkpoint_path, url])
        print("  Finish downloading checkpoint.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "generator_smooth" in checkpoint:
        generator.load_state_dict(checkpoint["generator_smooth"])
    else:
        generator.load_state_dict(checkpoint["generator"])
    # device = torch.device(3)  # changed to make it work Turso
    device = torch.device(device)  # changed back to wotk Turso
    # generator = generator.cuda()
    generator = generator.to(device)
    generator.eval()
    print("Finish loading checkpoint.")
    return generator


@torch.no_grad()
def tensor_from_pandas(frame_series: pd.DataFrame) -> torch.Tensor:
    is_series = isinstance(frame_series, pd.Series)
    series = frame_series if is_series else frame_series.squeeze()
    lat = torch.tensor(series).float()
    return torch.atleast_2d(lat)


@torch.no_grad()
def image_from_tensor(tensor: torch.Tensor) -> Image.Image:
    """Saves a tensor as an image to the specified path."""
    tensor = (tensor + 1) / 2  # Shift from [-1, 1] to [0, 1]
    tensor = tensor * 255
    tensor = tensor.clamp(0, 255)
    tensor = tensor.to(torch.uint8)
    return transforms.ToPILImage()(tensor)


def resize_tensor_image(image: torch.Tensor, *, size: int) -> torch.Tensor:
    """Resize the image tensor to a square of the given size."""
    resize_transform = transforms.Resize((size, size))
    return resize_transform(image)


def resize_image_pil(image: Image.Image, *, size: int) -> Image.Image:
    """Resize the PIL image represented to square size."""
    return image.resize((size, size))


def resize_image(
    input_data: Union[torch.Tensor, Image.Image],
    *,
    size: int,
) -> Union[torch.Tensor, Image.Image]:
    """Resize the input data to a square of the given size."""

    resize_function = MappingProxyType(
        {
            torch.Tensor: resize_tensor_image,
            Image.Image: resize_image_pil,
        }
    )

    for input_type, func in resize_function.items():
        if isinstance(input_data, input_type):
            return func(input_data, size=size)

    raise TypeError("Unsupported input data type")

face_generator = load_generator(
    model_name=_GEN_NAME, device=_DEV, checkpoints_dir="checkpoints"
)
    
def img_from_latent(latent, size=128):
    tensor = tensor_from_pandas(latent)
    face_generator.eval()
    out = face_generator(tensor)
    latent, image = out["z"].squeeze(), out["image"].squeeze()
    image = resize_image(image_from_tensor(image), size=size)
    return image


