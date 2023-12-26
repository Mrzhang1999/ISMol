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
import wandb
# 限制进程数
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
@ex.automain
def main(_config):
    # 初始化参数
    _config = copy.deepcopy(_config)

    # wandb.init(project="myCLIP", entity="zhxiang",settings=wandb.Settings(start_method="fork"))
    # 设置种子
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=False)

    model = myCLIP(_config)

    exp_name = f'{_config["exp_name"]}'
    # 日志打印文件
    os.makedirs(_config["log_dir"], exist_ok=True)
    # checkpoint保存配置
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",       # 想监视的指标
        mode=_config['mode'],
        save_last=False,
        dirpath=_config['checkpoint_save_path'],
        filename="{epoch:02d}-{global_step}-64",
    )

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     save_top_k=-1,
    #     monitor="epoch",
    #     mode="max",
    #     dirpath=_config['checkpoint_save_path'],
    #     filename="p1-{epoch:02d}-{global_step}-025",
    # )
    if not _config['test_only']:
        wandb_logger = WandbLogger(project=_config['log_project'])
    # wandb_logger = WandbLogger(project=_config['log_project'])
    # 学习率回调函数
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]
    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )
    # 4096 / (4*1*1)
    grad_steps = max(_config["batch_size"] // (
            _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    ), 1)
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    # 训练参数配置
    start_time = time.time()

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],  # 使用gpu列表
        num_nodes=_config["num_nodes"],  # 节点数
        precision=_config["precision"],  # 指定训练精度
        accelerator="cuda",  #
        benchmark=True,
        deterministic= not _config['is_pretrain'], # 预训练为False，用到了gather函数。微调用True,可复现
        max_epochs=_config["max_epoch"] if max_steps is None else 100,
        max_steps=max_steps,
        callbacks=callbacks,  # 回调函数,保存checkpoint
        logger= wandb_logger if not _config['test_only'] else None,  # 打印日志
        replace_sampler_ddp=False,  #
        accumulate_grad_batches=grad_steps,  # 每k次batches累计一次梯度
        log_every_n_steps=10,  # 更新n次网络权重后记录一次日志
        resume_from_checkpoint=_config["resume_from"],  #
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    # 训练
    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)

    # 测试
    else:
        trainer.test(model, datamodule=dm,ckpt_path=_config["load_path"])

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))