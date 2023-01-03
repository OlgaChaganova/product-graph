import os
import uuid

import pytorch_lightning as pl

from configs.base import (Callbacks, Common, Config, Dataset, LRScheduler,
                          Model, Optimizer, Project, Train)
from data.graph_dataset import OGBGProductsDatamodule
from models.gcn import GCNModule

RUN_NAME = 'gcn_' + uuid.uuid4().hex[:6]  # unique run id


CONFIG = Config(
    project=Project(
        log_freq=500,
        project_name='OGBN Product',
        run_name=RUN_NAME,
        tags='gcn',
        notes='',
    ),

    common=Common(seed=8),

    dataset=Dataset(
        module=OGBGProductsDatamodule,
        root='data/',
        batch_size=512,
        num_workers=6,
    ),

    model=Model(
        module=GCNModule,
        model_params={
            'num_features': [100, 256, 256, 256],
            'num_classes': 47,
            'dropout': 0.15,
        },
    ),

    train=Train(
        trainer_params={
            'devices': 1,
            'accelerator': 'auto',
            'accumulate_grad_batches': 1,
            'auto_scale_batch_size': None,
            'gradient_clip_val': 0.0,
            'benchmark': True,
            'precision': 32,
            'max_epochs': 20,
            'auto_lr_find': None,
        },

        callbacks=Callbacks(
            model_checkpoint=pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join('checkpoints', RUN_NAME),
                save_top_k=2,
                monitor='val_loss',
                mode='min',
            ),

            lr_monitor=pl.callbacks.LearningRateMonitor(logging_interval='step'),
        ),

        optimizer=Optimizer(
            name='Adam',
            opt_params={
                'lr': 0.001,
                'weight_decay': 0.0001,
            },
        ),

        lr_scheduler=LRScheduler(
            name='CosineAnnealingWarmRestarts',
            lr_sched_params={
                'T_0': 120,
                'T_mult': 1,
                'eta_min': 0.00001,
            },
        ),

        ckpt_path=None,
    ),
)