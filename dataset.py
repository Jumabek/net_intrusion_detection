import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from preprocessing import load_data, balance_data
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np

class MyDataset(Dataset):
    """
    A custom Dataset class for containing samples and their labels.
    """
    def __init__(self, X, y):
        """
        Initialize the dataset with input features X and labels y.

        Args:
            X (array-like): Input feature data.
            y (array-like): Corresponding labels for the input data.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    # def __getitem__(self, idx):
    #     return self.X[idx], self.y[idx]

class MyDataModule(pl.LightningDataModule):
    """
    Custom data module for handling data processing, split, and creating data loaders.
    """
    def __init__(self, dataroot, batch_size, seed, n_splits=5):
        """
        Initialize the data module with the data root directory, batch size, seed for reproducibility, 
        and number of splits for cross-validation.

        Args:
            dataroot (str): The root directory containing the data.
            batch_size (int): Batch size for DataLoader.
            seed (int): Seed for random number generation to ensure reproducibility.
            n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.
        """
        super().__init__()
        self.dataroot = dataroot
        self.batch_size = batch_size
        self.seed = seed
        self.n_splits = n_splits

    def prepare_data(self):
        # Load and preprocess data here, if necessary
        pass

    def setup(self, stage=None):
        X, y = load_data(self.dataroot)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        # Assuming you want the first fold for simplicity, you can change this logic to pick a specific fold
        train_val_indices, test_indices = next(skf.split(X, y))
        X_train_val, X_test = X[train_val_indices], X[test_indices]
        y_train_val, y_test = y[train_val_indices], y[test_indices]

        skf_inner = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        # Assuming you want the first inner fold for simplicity, you can change this logic to pick a specific fold
        train_indices, val_indices = next(skf_inner.split(X_train_val, y_train_val))
        X_train, X_val = X_train_val[train_indices], X_train_val[val_indices]
        y_train, y_val = y_train_val[train_indices], y_train_val[val_indices]

        self.train_dataset = MyDataset(X_train, y_train)
        self.val_dataset = MyDataset(X_val, y_val)
        self.test_dataset = MyDataset(X_test, y_test)
        
        input_sample, _ = self.train_dataset[0]
        self.input_dim = len(input_sample)
        self.output_dim = len(torch.unique(torch.tensor(y_train)))
    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value):
        self._input_dim = value

    @property
    def output_dim(self):
        return self._output_dim

    @output_dim.setter
    def output_dim(self, value):
        self._output_dim = value

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

