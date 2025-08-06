import os
import matplotlib.pyplot as plt
import numpy as np
from src.datasets import CityscapesDataset
from src.utils.process import process_item


def show_mask(dataset: CityscapesDataset, index: int=0):
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


def log_training(epoch, total_epochs, train_loss, test_loss, time):
    print(f"Epoch [{epoch}/{total_epochs}] - "
    f"Train Loss: {train_loss:.4f} - "
    f"Val Loss: {test_loss:.4f} - "
    f"Time: {time:.1f}s")


def plot_training_history(history, save_path):
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Validate Loss')
    plt.set_title('Loss')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()