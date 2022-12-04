import logging
import typing as tp

import pytorch_lightning as pl
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader


class OGBGProductsDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
    ):
        """
        Create Data Module for OGBG Product task.

        Parameters
        ----------
        root : str
            Path to the root directory.
        batch_size : int
            Batch size for dataloaders.
        num_workers : int
            Number of workers in dataloaders.
        """
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = PygNodePropPredDataset(name='ogbn-products', root=root)
        self.split_idx = self.dataset.get_idx_split()

    def setup(self, stage: tp.Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_split = self.split_idx['train']
            num_train_files = len(self.train_split)
            logging.info(f'Mode: train, number of files: {num_train_files}')

            self.valid_split = self.split_idx['valid']
            num_valid_files = len(self.valid_split)
            logging.info(f'Mode: valid, number of files: {num_valid_files}')

        elif stage == 'test':
            self.test_split = self.split_idx['test']
            num_test_files = len(self.test_split)
            logging.info(f'Mode: test, number of files: {num_test_files}')

    def train_dataloader(self):
        return DataLoader(
            self.dataset[self.train_split],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset[self.valid_split],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset[self.test_split],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
