import math

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup, AdamW

def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(
    txt_emb, img_emb, txt_pad, img_pad, beta=0.5, iteration=50, k=1
):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)

    T = ipot(
        cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta, iteration, k
    )
    distance = trace(cost.matmul(T.detach()))
    return distance

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    # layerNorm,仿射变换1,0
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    #
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    # head_names = [ "mlm_score", "itm_score", "mpp_score"]
    head_names = ["mlm_score","itm_score","decoder_block",'fpp_score','MoleculeNet_classify_score']
    cross_modal_names = ['cross_modal']
    lr_mult_head = pl_module.hparams.config["lr_mult_head"]
    lr_mult_cross_modal = pl_module.hparams.config["lr_mult_cross_modal"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay) # bias norm no-decay
                and not any(bb in n for bb in head_names) #
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,

        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult_head,

        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult_head,

        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult_cross_modal,

        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult_cross_modal,

        },
    ]
    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps is None:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )


def logger_mlm(pl_module, batch):
    images, _, smiles, *el= batch
    infer_mlm = pl_module.infer(images, smiles,mask_smiles = True)
    mlm_logits = pl_module.mlm_score(infer_mlm["smiles_feats"])
    mlm_labels = infer_mlm['mlm_labels']
    loss_mlm = pl_module.focal_loss(
        mlm_logits.view(-1, pl_module.config["vocab_size"]),
        mlm_labels.view(-1),
    )
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(loss_mlm)
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        mlm_logits, mlm_labels
    )
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)
    print(f"mlm accuracy:{acc}")
    return {
        "loss_mlm": loss_mlm,
        "acc_mlm": acc,
    }


def logger_itm(pl_module, batch):
    images,image_false,smiles,*el = batch

    pos_len = images.size(0) // 2
    neg_len = images.size(0) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(images.device)

    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]
    itm_images = torch.stack(
        [images[idx] if label == 1 else image_false[idx] for idx, label in enumerate(itm_labels)]).to(images.device)
    infer_itm = pl_module.infer(itm_images, smiles)
    itm_logits = pl_module.itm_score(infer_itm["cls_feats"])

    loss_itm = F.cross_entropy(itm_logits, itm_labels.long())

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(loss_itm)  # Metrics.forwrad()
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        itm_logits, itm_labels
    )
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)
    return {
        "loss_itm": 8 * loss_itm,
        "acc_itm": acc,
    }


def logger_fpp(pl_module, batch):
    images, _, smiles, _ , k_100,k_1000,k_10000 = batch

    infer_fpp = pl_module.infer(images, smiles)

    fpp_logits_image100 = pl_module.fpp_score_images100(infer_fpp['cls_feats_image'])
    loss_fpp_image100 = pl_module.focal_loss(
        fpp_logits_image100.view(-1, 100),
        k_100.view(-1),
    )
    fpp_logits_smiles100 = pl_module.fpp_score_smiles100(infer_fpp['cls_feats_smiles'])
    loss_fpp_smiles100 = F.cross_entropy(
        fpp_logits_smiles100.view(-1, 100),
        k_100.view(-1),
    )

    fpp_logits_image1000 = pl_module.fpp_score_images1000(infer_fpp['cls_feats_image'])
    loss_fpp_image1000 = pl_module.focal_loss(
        fpp_logits_image1000.view(-1, 1000),
        k_1000.view(-1),
    )
    fpp_logits_smiles1000 = pl_module.fpp_score_smiles1000(infer_fpp['cls_feats_smiles'])
    loss_fpp_smiles1000 = F.cross_entropy(
        fpp_logits_smiles1000.view(-1, 1000),
        k_1000.view(-1),
    )

    fpp_logits_image10000 = pl_module.fpp_score_images10000(infer_fpp['cls_feats_image'])
    loss_fpp_image10000 = pl_module.focal_loss(
        fpp_logits_image10000.view(-1, 10000),
        k_10000.view(-1),
    )
    fpp_logits_smiles10000 = pl_module.fpp_score_smiles10000(infer_fpp['cls_feats_smiles'])
    loss_fpp_smiles10000 = F.cross_entropy(
        fpp_logits_smiles10000.view(-1, 10000),
        k_10000.view(-1),
    )

    total_loss = loss_fpp_image100 + loss_fpp_smiles100 + loss_fpp_image1000 + loss_fpp_smiles1000 + loss_fpp_image10000 + loss_fpp_smiles10000


    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_fpp_loss")(total_loss)

    acc_image100 = getattr(pl_module, f"{phase}_fpp_accuracy")(
        fpp_logits_image100, k_100,k = True,
    )
    acc_smiles100 = getattr(pl_module, f"{phase}_fpp_accuracy")(
        fpp_logits_smiles100, k_100,k = True,
    )
    acc_image1000 = getattr(pl_module, f"{phase}_fpp_accuracy")(
        fpp_logits_image1000, k_1000,k = True,
    )
    acc_smiles1000 = getattr(pl_module, f"{phase}_fpp_accuracy")(
        fpp_logits_smiles1000, k_1000,k = True,
    )
    acc_image10000 = getattr(pl_module, f"{phase}_fpp_accuracy")(
        fpp_logits_image10000, k_10000,k = True,
    )
    acc_smiles10000 = getattr(pl_module, f"{phase}_fpp_accuracy")(
        fpp_logits_smiles10000, k_10000,k = True,
    )
    pl_module.log(f"fpp/{phase}/loss", loss)
    pl_module.log(f"fpp/{phase}/accuracy", (acc_image100 + acc_smiles100 + acc_image1000 + acc_smiles1000 + acc_image10000 + acc_smiles10000)/6)
    print(f'acc_image100: {acc_image100},acc_smiles100:{acc_smiles100}, acc_image1000:{acc_image1000} , '
          f'acc_smiles1000:{acc_smiles1000}, acc_image10000:{acc_image10000}, acc_smiles10000:{acc_smiles10000}')
    return {
        "loss_fpp": total_loss,
        "acc_fpp_smiles100": acc_smiles100,
        "acc_fpp_image100": acc_image100,
        "acc_fpp_smiles1000": acc_smiles1000,
        "acc_fpp_image1000": acc_image1000,
        "acc_fpp_smiles10000": acc_smiles10000,
        "acc_fpp_image10000": acc_image10000,
    }

def compute_MoleculeNet_classify(pl_module, batch,testing=False):
    smiles,images,label = batch
    infer_MoleculeNet_classify = pl_module.infer(images, smiles)
    MoleculeNet_classify_logits = pl_module.MoleculeNet_classify_score(infer_MoleculeNet_classify['cls_feats'])
    loss_MoleculeNet_classify = pl_module.focal_loss(
        MoleculeNet_classify_logits.view(-1, 2),
        label.view(-1),
    )
    phase = "train" if pl_module.training else "val"
    if testing == True:
        phase = "test"
    loss = getattr(pl_module, f"{phase}_MoleculeNet_classify_loss")(loss_MoleculeNet_classify)
    acc = getattr(pl_module, f"{phase}_MoleculeNet_classify_accuracy")(
        MoleculeNet_classify_logits, label,k = True,
    )
    auroc = getattr(pl_module,f"{phase}_MoleculeNet_classify_auroc")(
        MoleculeNet_classify_logits, label
    )
    pl_module.log(f"MoleculeNet_classify/{phase}/loss", loss)
    pl_module.log(f"MoleculeNet_classify/{phase}/accuracy", acc)
    pl_module.log(f"MoleculeNet_classify/{phase}/auroc",auroc)

    return {
        "loss_MoleculeNet_classify": loss_MoleculeNet_classify,
        "acc_MoleculeNet_classify": acc,
        'auroc': auroc,
        'logits': MoleculeNet_classify_logits,
    }

class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T
