import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from preprocessing import load_data, balance_data


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MyDataModule(pl.LightningDataModule):
    def __init__(self, dataroot, batch_size, seed, val_split=0.1, test_split=0.1):
        super().__init__()
        self.dataroot = dataroot
        self.batch_size = batch_size
        self.seed = seed
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        X, y = load_data(self.dataroot)
        X, y = balance_data(X, y, self.seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_split, random_state=self.seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.val_split, random_state=self.seed)
        self.train_dataset = MyDataset(X_train, y_train)
        self.val_dataset = MyDataset(X_val, y_val)
        self.test_dataset = MyDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )

