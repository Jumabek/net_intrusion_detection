import argparse

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset import MyDataModule


def main(args):
    # Initialize data module
    data_module = MyDataModule(
        dataroot=args.dataroot,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    data_module.prepare_data()
    data_module.setup()

    # Print dataset shapes
    print(f"Train inputs shape: {data_module.train_dataset.X.shape}")
    print(f"Train targets shape: {data_module.train_dataset.y.shape}")
    print(f"Validation inputs shape: {data_module.val_dataset.X.shape}")
    print(f"Validation targets shape: {data_module.val_dataset.y.shape}")
    print(f"Test inputs shape: {data_module.test_dataset.X.shape}")
    print(f"Test targets shape: {data_module.test_dataset.y.shape}")

    # Initialize data loader
    dataloader = DataLoader(
        data_module.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Print batch shapes
    inputs, targets = next(iter(dataloader))
    print(f"Batch inputs shape: {inputs.shape}")
    print(f"Batch targets shape: {targets.shape}")

# python test_datamodule.py MachineLearningCVE/
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MyDataModule")
    parser.add_argument("dataroot", type=str, help="path to dataset root directory")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size for training")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    main(args)
