import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm


class Trainer():
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 device: torch.device,
                 epochs: int = 10):
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
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(imgs)

            output_img = torch.Tensor(output[0])
            output_mask = torch.Tensor(output[1])

            loss = self.criterion(output_img, output_mask)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)

        total_loss = epoch_loss / len(self.train_loader.dataset)
        return total_loss

    def _validate_epoch(self):
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for imgs, masks in tqdm(self.val_loader, desc="Validation", leave=False):
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                output = self.model(imgs)
                loss = self.criterion(output, masks)
                epoch_loss += loss.item() * imgs.size(0)

        total_loss = epoch_loss / len(self.val_loader.dataset)
        return total_loss

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()

            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()

            elapsed = time.time() - start_time

            # TODO: сохранение модели

            print(f"Epoch [{epoch}/{self.epochs}] - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Time: {elapsed:.1f}s")
