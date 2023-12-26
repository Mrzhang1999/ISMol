import math

import torch
import torchmetrics

from torchmetrics import Metric
from sklearn.metrics import roc_curve


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target,k=False):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        preds = logits.argmax(dim=-1)
        if not k:
            preds = preds[target != -100]
            target = target[target != -100]
            if target.numel() == 0:
                return 1

            assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total

class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total


class ROAUC(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("logits", default=list(), dist_reduce_fx="cat")
        self.add_state("target", default=list(), dist_reduce_fx="cat")
        self.roauc = torchmetrics.AUROC(num_classes=2)

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.roauc.device),
            target.detach().to(self.roauc.device),
        )
        self.logits.append(logits)
        self.target.append(target)

    def compute(self):
        logits = torch.cat(self.logits,dim=0)
        target = torch.cat(self.target,dim=0)
        return self.roauc(logits,target)
