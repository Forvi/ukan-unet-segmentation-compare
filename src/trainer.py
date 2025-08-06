import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

from src.utils.visualize import log_training


class Trainer():
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 device: torch.device,
                 epochs: int=10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs

    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0.0

        for imgs, masks in tqdm(self.train_loader, desc="Training", leave=False):
            self.optimizer.zero_grad()

            print(f'IMAGE SHAPE: {imgs.shape}')
            print(f'MASK SHAPE: {masks.shape}')

            imgs = imgs.to(self.device)
            masks = masks.to(self.device).long().squeeze(1)

            output = self.model(imgs)
            loss = self.criterion(output[1], masks)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            print(f'LOSS: {loss.item()}')

        total_loss = epoch_loss / len(self.train_loader)
        print(f'RESULT TRAINING LOSS: {total_loss}')
        return total_loss

    def _validate_epoch(self):
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for imgs, masks in tqdm(self.val_loader, desc="Validation", leave=False):
                print(f'IMAGE SHAPE: {imgs.shape}')
                print(f'MASK SHAPE: {masks.shape}')

                imgs = imgs.to(self.device)
                masks = masks.to(self.device).long().squeeze(1)
                output = self.model(imgs)
                loss = self.criterion(output[1], masks)
                epoch_loss += loss.item()

        total_loss = epoch_loss / len(self.val_loader)
        print(f'RESULT TEST LOSS: {total_loss}')
        return total_loss

    def fit(self):
        train_losses = []
        val_losses = []

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()

            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()

            end_time = time.time() - start_time

            # TODO: сохранение модели

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            log_training(epoch, self.epochs, train_loss, val_loss, end_time)

        return {
            'train_losses' : train_losses,
            'val_losses': val_losses
        }

"""
размеры в train:
IMAGE SHAPE: torch.Size([16, 3, 256, 512])
MASK SHAPE: torch.Size([16, 1, 256, 512])

размеры в val:
IMAGE SHAPE: torch.Size([16, 3, 256, 512])
MASK SHAPE: torch.Size([16, 1, 256, 512])

"""