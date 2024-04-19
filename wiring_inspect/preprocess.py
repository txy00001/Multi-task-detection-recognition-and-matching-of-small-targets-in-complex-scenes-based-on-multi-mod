from PIL import Image, ImageFilter
import torch
import numpy as np


def dilate(mask: Image):
    return mask.filter(ImageFilter.MaxFilter(19))


def dilate_torch(mask: Image):
    mask = torch.from_numpy(np.asarray(mask)).to("cuda").unsqueeze(0).unsqueeze(0).float()
    mask = torch.nn.functional.max_pool2d(mask, 19, 1, 9)
    return Image.fromarray(mask.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8))


def rotate(crop: Image, angle: float):
    return crop.rotate(angle, expand=True)


PREPROCESS_MAP = {
    "dilate": dilate_torch,
    "rotate": rotate,
}