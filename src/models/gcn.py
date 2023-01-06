import logging
import typing as tp

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from torchmetrics import Accuracy, F1Score, Precision, Recall
from tqdm import tqdm

from configs.base import LRScheduler, Optimizer


class GraphConvNetwork(nn.Module):
    """Graph Convolutional Network."""

    def __init__(
        self,
        num_features: tp.List[int],
        num_classes: int,
        dropout: float
    ):
        """
        Initialize GraphConvNetwork.

        Parameters
        ----------
        num_features : tp.List[int]
            Number of features in each vertex (number of convolutions).
        num_classes : int
            Number of classes in data.
        dropout : float, optional
           Dropout rate, by default 0.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        num_layers = len(num_features)
        for i in range(num_layers - 1):
            self.layers.append(
                SAGEConv(in_channels=num_features[i], out_channels=num_features[i + 1]),
            )
        self.layers.append(
            SAGEConv(in_channels=num_features[-1], out_channels=num_classes),
        )
        self.dropout = dropout

    def forward(self, x: torch.tensor, edge_index: torch.tensor) -> torch.tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        return torch.log_softmax(x, dim=-1)

    def inference(self, x_all: torch.tensor, subgraph_loader: NeighborSampler) -> torch.tensor:
        for i, conv in enumerate(self.layers):
            conv = conv.cpu()
            xs = []
            for batch_size, n_id, adj in tqdm(subgraph_loader):
                edge_index, _, size = adj
                x = x_all[n_id]
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.layers) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
        return x_all


class GCNModule(pl.LightningModule):
    def __init__(
        self,
        model_params: dict,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None,
    ):
        """
        Initialize GCNModule.

        Parameters
        ----------
        model_params : dict
            Dictionary with GraphConvNetwork parameters.
        optimizer : Optimizer
            Optimizer.
        lr_scheduler : LRScheduler
            Learning rate scheduler.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['criterion'])  # criterion is already saved during checkpointing
        self.learning_rate = optimizer.opt_params['lr']

        self.model = GraphConvNetwork(**model_params)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.criterion = nn.CrossEntropyLoss()

        num_classes = model_params['num_classes']
        task = 'multiclass'
        self.accuracy = Accuracy(task=task, num_classes=num_classes)
        self.f1_score = F1Score(task=task, num_classes=num_classes)
        self.precision_score = Precision(task=task, num_classes=num_classes)
        self.recall_score = Recall(task=task, num_classes=num_classes)

        self.test_probs = None

    def forward(self, x: torch.tensor, edge_index: torch.tensor):
        return self.model(x, edge_index)

    def training_step(self, batch, batch_idx):
        targets = batch.y.squeeze(1)[batch.train_mask]
        probs = self.forward(batch.x, batch.edge_index)[batch.train_mask]
        batch_size = len(targets)

        loss = self.criterion(probs, targets)
        self.log('train_loss', loss, on_epoch=False, on_step=True, batch_size=batch_size)

        self.accuracy(probs, targets.long())
        self.log('train_acc', self.accuracy, on_epoch=False, on_step=True, batch_size=batch_size)

        self.f1_score(probs, targets.long())
        self.log('train_f1', self.f1_score, on_epoch=False, on_step=True, batch_size=batch_size)

        self.precision_score(probs, targets.long())
        self.log('train_precision', self.precision_score, on_epoch=False, on_step=True, batch_size=batch_size)

        self.recall_score(probs, targets.long())
        self.log('train_recall_score', self.recall_score, on_epoch=False, on_step=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        targets = batch.y.squeeze(1)[batch.valid_mask]
        probs = self.forward(batch.x, batch.edge_index)[batch.valid_mask]
        batch_size = len(targets)

        loss = self.criterion(probs, targets)
        self.log('val_loss', loss, on_epoch=True, on_step=True, batch_size=batch_size)

        self.accuracy(probs, targets.long())
        self.log('val_acc', self.accuracy, on_epoch=True, on_step=False, batch_size=batch_size)

        self.f1_score(probs, targets.long())
        self.log('val_f1', self.f1_score, on_epoch=True, on_step=False, batch_size=batch_size)

        self.precision_score(probs, targets.long())
        self.log('val_precision', self.precision_score, on_epoch=True, on_step=False, batch_size=batch_size)

        self.recall_score(probs, targets.long())
        self.log('val_recall_score', self.recall_score, on_epoch=True, on_step=False, batch_size=batch_size)
        return loss

    def on_test_start(self) -> None:
        logging.info('Starting testing...')
        self.test_probs = self.model.inference(
            x_all=self.trainer.datamodule.data.x,
            subgraph_loader=self.trainer.test_dataloaders[0],
        )

    def test_step(self, batch, batch_idx):
        return 0

    def on_test_end(self) -> None:
        targets = self.trainer.datamodule.data.y
        y_pred = self.test_probs.argmax(dim=-1, keepdim=True)

        acc = self.accuracy(y_pred, targets.long())
        f1_score = self.f1_score(y_pred, targets.long())
        precision_score = self.precision_score(y_pred, targets.long())
        recall_score = self.recall_score(y_pred, targets.long())

        self.logger.experiment.log(
            {
                'test_acc': acc,
                'test_f1': f1_score,
                'test_precision': precision_score,
                'test_recall_score': recall_score,
            }
        )

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer.name)(
            self.parameters(),
            **self.optimizer.opt_params,
        )

        optim_dict = {
            'optimizer': optimizer,
            'monitor': 'val_loss',
        }

        if self.lr_scheduler is not None:
            lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler.name)(
                optimizer,
                **self.lr_scheduler.lr_sched_params,
            )

            optim_dict.update({'lr_scheduler': lr_scheduler})
        return optim_dict
