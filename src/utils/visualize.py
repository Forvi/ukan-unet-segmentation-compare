import matplotlib.pyplot as plt
import numpy as np
from src.datasets import CityscapesDataset
from src.utils.process import process_item


def show_mask(dataset: CityscapesDataset, index: int = 0):
    img, mask = dataset.__getitem__(index)
    img, mask = process_item(img, mask)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(img.astype(np.uint8))
    axs[0].set_title('Image')
    axs[0].axis('off')

    axs[1].imshow(mask)
    axs[1].set_title('Mask')
    axs[1].axis('off')

    plt.show()


