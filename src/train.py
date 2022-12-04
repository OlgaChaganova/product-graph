import argparse
import logging
import os
import typing as tp
from runpy import run_path

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from configs.base import Config
from configs.utils import get_config_dict


def get_wandb_logger(model: pl.LightningModule, datamodule: pl.LightningDataModule, config: Config) -> WandbLogger:
    config_dict = get_config_dict(model=model, datamodule=datamodule, config=config)
    wandb_logger = WandbLogger(
        project=config.project.project_name,
        name=config.project.run_name,
        tags=config.project.tags.split(','),
        notes=config.project.notes,
        config=config_dict,
        log_model='all',
    )
    wandb_logger.watch(
        model=model,
        log='all',
        log_freq=config.project.log_freq,  # log gradients and parameters every log_freq batches
    )
    return wandb_logger


def parse() -> tp.Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=os.path.join('src', 'configs', 'config_mlp.py'),
        type=str,
        help='Path to experiment config file (*.py)',
    )
    return parser.parse_args()


def main(args: tp.Any, config: Config):
    # model
    model = config.model.module(
        config.model.model_params,
        optimizer=config.train.optimizer,
        lr_scheduler=config.train.lr_scheduler,
    )

    # data module
    datamodule = config.dataset.module(
        root=config.dataset.root,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
    )

    # logger
    logger = get_wandb_logger(model, datamodule, config)

    # trainer
    trainer_params = config.train.trainer_params
    callbacks = list(config.train.callbacks.__dict__.values())
    callbacks = filter(lambda callback: callback is not None, callbacks)
    trainer = Trainer(
        callbacks=[
            TQDMProgressBar(refresh_rate=1),
            RichModelSummary(),
            *callbacks,
        ],
        logger=logger,
        **trainer_params,
    )

    if trainer_params['auto_scale_batch_size'] is not None or trainer_params['auto_lr_find'] is not None:
        trainer.tune(model=model, datamodule=datamodule)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=config.train.ckpt_path,
    )

    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path='best',
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse()
    config_module = run_path(args.config)
    exp_config = config_module['CONFIG']
    seed_everything(exp_config.common.seed, workers=True)
    main(args, exp_config)
