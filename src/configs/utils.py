import pytorch_lightning as pl

from configs.base import Config, LRScheduler, Optimizer

SEP = '_'


def get_dict_for_optimizer(config_optimizer: Optimizer) -> dict:
    opt_dict = {}
    opt_dict[SEP.join(['optimizer', 'name'])] = config_optimizer.name
    for key, value in config_optimizer.opt_params.items():
        opt_dict[SEP.join(['optimizer', key])] = value
    return opt_dict


def get_dict_for_lr_scheduler(config_lr_scheduler: LRScheduler) -> dict:
    lr_dict = {}
    lr_dict[SEP.join(['lr_scheduler', 'name'])] = config_lr_scheduler.name
    for key, value in config_lr_scheduler.lr_sched_params.items():
        lr_dict[SEP.join(['lr_scheduler', key])] = value
    return lr_dict


def get_config_dict(model: pl.LightningModule, datamodule: pl.LightningDataModule, config: Config) -> dict:
    config_dict = {}
    datamodule_dict = dict(datamodule.hparams)
    trainer_dict = config.train.trainer_params
    optimizer_dict = get_dict_for_optimizer(config.train.optimizer)
    lr_sched_dict = get_dict_for_lr_scheduler(config.train.lr_scheduler)

    # using update to avoid doubling keys
    # config_dict.update(model_dict)
    config_dict.update(datamodule_dict)
    config_dict.update(trainer_dict)
    config_dict.update(optimizer_dict)
    config_dict.update(lr_sched_dict)
    return config_dict
