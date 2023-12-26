import os
import numpy as np
import random
import time
import datetime
import torch
import copy
from config import ex
import pytorch_lightning as pl
from datamodules.datamodules_multitask import MTDataModule
from models.myCLIP import myCLIP

from pytorch_lightning.loggers import WandbLogger
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
@ex.automain
def main(_config):

    _config = copy.deepcopy(_config)

    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=False)

    model = myCLIP(_config)

    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode=_config['mode'],
        save_last=False,
        dirpath=_config['checkpoint_save_path'],
        filename="{epoch:02d}-{global_step}-64",
    )

    if not _config['test_only']:
        wandb_logger = WandbLogger(project=_config['log_project'])
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]
    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = max(_config["batch_size"] // (
            _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    ), 1)
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    start_time = time.time()

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="cuda",
        benchmark=True,
        deterministic= not _config['is_pretrain'],
        max_epochs=_config["max_epoch"] if max_steps is None else 100,
        max_steps=max_steps,
        callbacks=callbacks,
        logger= wandb_logger if not _config['test_only'] else None,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm,ckpt_path=_config["load_path"])

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))