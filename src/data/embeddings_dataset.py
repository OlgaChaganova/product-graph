import logging
import typing as tp

import pytorch_lightning as pl
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch.utils.data import DataLoader, Dataset


class EmbeddingsDataset(Dataset):
    def __init__(
        self,
        root: str,
        mode: tp.Literal['train', 'valid', 'test'],
    ):
        self.root = root
        embeddings, targets = self._get_embeddings_and_targets(mode)
        self.embeddings = embeddings
        self.targets = targets

    def _get_embeddings_and_targets(self, mode: str) -> tp.Tuple[torch.tensor, torch.tensor]:
        dataset = PygNodePropPredDataset(name='ogbn-products', root=self.root)
        graph = dataset[0]
        indices = dataset.get_idx_split()[mode]
        embeddings = graph.x[indices]
        targets = graph.y[indices]
        return embeddings, targets

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        embedding = self.embeddings[index]
        target = self.targets[index]
        return embedding, target


class EmbeddingsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
    ):
        """
        Create Data Module for EmbeddingsDataset.

        Parameters
        ----------
        root : str
            Path to root dir with dataset.
        batch_size : int
            Batch size for dataloaders.
        num_workers : int
            Number of workers in dataloaders.
        """
        super().__init__()
        self.save_hyperparameters()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: tp.Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = EmbeddingsDataset(
                root=self.root,
                mode='train',
            )
            num_train_files = len(self.train_dataset)
            logging.info(f'Mode: train, number of files: {num_train_files}')

            self.val_dataset = EmbeddingsDataset(
                root=self.root,
                mode='valid',
            )
            num_val_files = len(self.val_dataset)
            logging.info(f'Mode: val, number of files: {num_val_files}')

        elif stage == 'test':
            self.test_dataset = EmbeddingsDataset(
                root=self.root,
                mode='test',
            )
            num_test_files = len(self.test_dataset)
            logging.info(f'Mode: test, number of files: {num_test_files}')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False,
        )
