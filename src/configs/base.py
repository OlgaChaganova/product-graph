import typing as tp
from dataclasses import dataclass

import pytorch_lightning as pl


@dataclass
class Project:
    project_name: str
    run_name: str
    notes: str
    tags: str
    log_freq: int = 100


@dataclass
class Common:
    seed: int = 8


@dataclass
class Dataset:
    module: pl.LightningDataModule
    root: str  # path to root directory with images
    batch_size: int
    num_workers: int


@dataclass
class Model:
    module: pl.LightningModule
    model_params: dict


@dataclass
class Callbacks:
    model_checkpoint: pl.callbacks.ModelCheckpoint
    early_stopping: tp.Optional[pl.callbacks.EarlyStopping] = None
    lr_monitor: tp.Optional[pl.callbacks.LearningRateMonitor] = None
    model_summary: tp.Optional[tp.Union[pl.callbacks.ModelSummary, pl.callbacks.RichModelSummary]] = None
    timer: tp.Optional[pl.callbacks.Timer] = None


@dataclass
class Optimizer:
    name: str
    opt_params: dict


@dataclass
class LRScheduler:
    name: str
    lr_sched_params: dict


@dataclass
class Train:
    trainer_params: dict
    callbacks: Callbacks
    optimizer: Optimizer
    lr_scheduler: LRScheduler
    ckpt_path: tp.Optional[str] = None


@dataclass
class Config:
    project: Project
    common: Common
    dataset: Dataset
    model: Model
    train: Train
