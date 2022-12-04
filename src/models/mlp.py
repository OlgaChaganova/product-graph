import typing as tp

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score, Precision, Recall

from configs.base import LRScheduler, Optimizer


class MultiLayerPerceptron(nn.Module):
    """Base MLP model using here as a baseline."""

    def __init__(
        self,
        num_neurons: tp.List[int],
        num_classes: int,
        dropout: float = 0,
    ):
        """
        Initialize MultiLayerPerceptron.

        Parameters
        ----------
        num_neurons : tp.List[int]
            Number of neurons in each layer.
        num_classes : int
            Number of classes in data.
        dropout : float, optional
           Dropout rate, by default 0.
        """
        super().__init__()

        self.lin_layers = []
        num_layers = len(num_neurons)
        for i in range(num_layers - 1):
            self.lin_layers.extend(
                [
                    nn.Linear(num_neurons[i], num_neurons[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ],
            )
        self.lin_layers.append(
            nn.Linear(num_neurons[-1], num_classes),
        )
        self.model = nn.Sequential(*self.lin_layers)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.model(x)
        return torch.log_softmax(x, dim=-1)


class MLPModule(pl.LightningModule):
    def __init__(
        self,
        model_params: dict,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None,
    ):
        """
        Initialize MLPModule.

        Parameters
        ----------
        model_params : dict
            Dictionary with MultiLayerPerceptron parameters.
        optimizer : Optimizer
            Optimizer.
        lr_scheduler : LRScheduler
            Learning rate scheduler.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['criterion'])  # criterion is already saved during checkpointing
        self.learning_rate = optimizer.opt_params['lr']

        self.model = MultiLayerPerceptron(**model_params)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.criterion = nn.CrossEntropyLoss()

        num_classes = model_params['num_classes']
        task = 'multiclass'
        self.accuracy = Accuracy(task=task, num_classes=num_classes)
        self.f1_score = F1Score(task=task, num_classes=num_classes)
        self.precision_score = Precision(task=task, num_classes=num_classes)
        self.recall_score = Recall(task=task, num_classes=num_classes)

    def forward(self, x: torch.tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, targets = batch
        targets = targets.squeeze(1)
        probs = self.forward(x)

        loss = self.criterion(probs, targets)
        self.log('train_loss', loss, on_epoch=False, on_step=True)

        self.accuracy(probs, targets.long())
        self.log('train_acc', self.accuracy, on_epoch=False, on_step=True)

        self.f1_score(probs, targets.long())
        self.log('train_f1', self.f1_score, on_epoch=False, on_step=True)

        self.precision_score(probs, targets.long())
        self.log('train_precision', self.precision_score, on_epoch=False, on_step=True)

        self.recall_score(probs, targets.long())
        self.log('train_recall_score', self.recall_score, on_epoch=False, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        targets = targets.squeeze(1)
        probs = self.forward(x)

        loss = self.criterion(probs, targets)
        self.log('val_loss', loss, on_epoch=True, on_step=True)

        self.accuracy(probs, targets.long())
        self.log('val_acc', self.accuracy, on_epoch=True, on_step=False)

        self.f1_score(probs, targets.long())
        self.log('val_f1', self.f1_score, on_epoch=True, on_step=False)

        self.precision_score(probs, targets.long())
        self.log('val_precision', self.precision_score, on_epoch=True, on_step=False)

        self.recall_score(probs, targets.long())
        self.log('val_recall_score', self.recall_score, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, targets = batch
        targets = targets.squeeze(1)
        probs = self.forward(x)

        loss = self.criterion(probs, targets)
        self.log('test_loss', loss)

        self.accuracy(probs, targets.long())
        self.log('test_acc', self.accuracy)

        self.f1_score(probs, targets.long())
        self.log('test_f1', self.f1_score)

        self.precision_score(probs, targets.long())
        self.log('test_precision', self.precision_score)

        self.recall_score(probs, targets.long())
        self.log('test_recall_score', self.recall_score)
        return loss

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
