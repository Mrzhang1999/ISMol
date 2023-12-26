import torch
import random

from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from evaluate.metrics import  Scalar, Accuracy, ROAUC


def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v <= 0:
                continue
            if k == "itc":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "itm" or k == "mlm":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k=='mpp':
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k=='MoleculeNet_classify':
                if split=='train':
                    setattr(pl_module,f"{split}_{k}_auroc", ROAUC())
                    setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"{split}_{k}_loss", Scalar())
                else:
                    setattr(pl_module, f"{split}_{k}_auroc", ROAUC())
                    setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"{split}_{k}_loss", Scalar())

                    setattr(pl_module, f"test_{k}_auroc", ROAUC())
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"test_{k}_loss", Scalar())
            elif k=='MoleculeNet_regress':
                # loss
                if split=='train':
                    setattr(pl_module, f"{split}_{k}_loss", Scalar())
                else:
                    setattr(pl_module, f"{split}_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_loss", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    the_metric = 0
    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v <= 0:
            continue
        value = 0
        if loss_name == "itm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "mpp":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        # 两个下游Molecule 分类回归任务
        elif loss_name == "MoleculeNet_classify":
            # 监测AUROC
            value = getattr(pl_module, f"{phase}_{loss_name}_auroc").compute()
            pl_module.log(f"{loss_name}/{phase}/auroc_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_auroc").reset()

            # acc
            pl_module.log(
                f"{loss_name}/{phase}/accuracy_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            )
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()

            # loss
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        elif loss_name == "MoleculeNet_regress":
            # 监测RMSE,需要把损失设置为 RMSE
            value = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            pl_module.log(f"{loss_name}/{phase}/loss", value)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        else:   # fpp、mlm、ocsr
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        the_metric += value

    # 多任务时监测精度和，当下游单任务时，即为单任务的监测目标
    pl_module.log(f"{phase}/the_metric", the_metric)

def test_epoch_auroc(pl_module):
    test_metric = 0
    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v <= 0:
            continue
        value = 0
        if loss_name == "MoleculeNet_classify":
            # 监测AUROC
            value = getattr(pl_module, f"test_{loss_name}_auroc").compute()
            pl_module.log(f"{loss_name}/test/auroc_epoch", value)
            getattr(pl_module, f"test_{loss_name}_auroc").reset()

            # acc
            pl_module.log(
                f"{loss_name}/test/accuracy_epoch",
                getattr(pl_module, f"test_{loss_name}_accuracy").compute()
            )
            getattr(pl_module, f"test_{loss_name}_accuracy").reset()

            # loss
            pl_module.log(
                f"{loss_name}/test/loss_epoch",
                getattr(pl_module, f"test_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"test_{loss_name}_loss").reset()

        elif loss_name == "MoleculeNet_regress":
            # 监测RMSE, 需要把损失设置为 RMSE
            value = getattr(pl_module, f"test_{loss_name}_rmse").compute()
            pl_module.log(f"{loss_name}_test_loss", value)
            getattr(pl_module, f"test_{loss_name}_loss").reset()
        else:
            phase = "test"
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        test_metric += value
    # 多任务时监测精度和，当下游单任务时，即为单任务的监测目标
    pl_module.log("test/the_metric", test_metric)

def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()

def set_task(pl_module):
    # 设置task name
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v > 0
    ]
    return