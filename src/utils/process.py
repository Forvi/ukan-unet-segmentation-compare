import numpy as np


def process_item(img: np.array, mask: np.array) -> np.array:
    img_np = np.array(img)
    mask_np = np.array(mask)
    return img_np, mask_np