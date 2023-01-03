import logging
import typing as tp

import torch
import pytorch_lightning as pl
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler


class OGBGProductsDatamodule(pl.LightningDataModule):
    """DataModule for mini-batch GCN training using the Cluster-GCN algorithm."""
    
    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        save_dir: tp.Optional[str] = 'data/ogbn_products/processed',
        num_partitions: int = 15000,
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
        save_dir : tp.Optional[str]
            Directory where already partitioned dataset is stored.
        num_partitions : int
            Number of partitions.
        """
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root
        self.num_partitions = num_partitions
        self.save_dir = save_dir

        self.data = None
        self.split_idx = None
        self.cluster_data = None
    
    def prepare_data(self):
        dataset = PygNodePropPredDataset(name='ogbn-products', root=self.root)
        self.split_idx = dataset.get_idx_split()
        self.data = dataset[0]

        # Convert split indices to boolean masks and add them to `data`
        for key, idx in self.split_idx.items():
            mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            self.data[f'{key}_mask'] = mask

        self.cluster_data = ClusterData(
            self.data,
            num_parts=self.num_partitions,
            recursive=False,
            save_dir=self.save_dir,
        )

    def setup(self, stage: tp.Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_split = self.split_idx['train']
            num_train_files = len(self.train_split)
            logging.info(f'Mode: train, number of nodes: {num_train_files}')

            self.valid_split = self.split_idx['valid']
            num_valid_files = len(self.valid_split)
            logging.info(f'Mode: valid, number of nodes: {num_valid_files}')

        elif stage == 'test':
            self.test_split = self.split_idx['test']
            num_test_files = len(self.test_split)
            logging.info(f'Mode: test, number of nodes: {num_test_files}')

    def train_dataloader(self):
        return ClusterLoader(
            self.cluster_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return ClusterLoader(
            self.cluster_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return NeighborSampler(
            self.data.edge_index,
            sizes=[-1],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
