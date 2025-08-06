from src.datasets import CityscapesDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch

from src.models import UNetFormerModel
from src.trainer import Trainer
from src.utils.visualize import plot_training_history


if __name__ == '__main__':
    path = './data/Cityscapes'
    train_dataset, test_dataset = CityscapesDataset.get_loaders(root=path, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNetFormerModel(num_classes=34)
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(model=model,
                      train_loader=train_dataset,
                      val_loader=test_dataset,
                      optimizer=optimizer,
                      criterion=criterion,
                      device=device,
                      epochs=10)

    history = trainer.fit()
    plot_training_history(history=history, save_path='./plots/result.png')

