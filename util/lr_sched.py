# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from torch.optim.lr_scheduler import LRScheduler

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

class CustomCosineWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, total_steps, min_lr, base_lr, last_epoch=-1):
        """
        Custom scheduler that implements warmup followed by half-cycle cosine decay
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of epochs for linear warmup
            total_epochs: Total number of training epochs
            min_lr: Minimum learning rate at the end of training
            base_lr: Base learning rate after warmup
            last_epoch: The index of last epoch
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.step_per_epoch = total_steps / total_epochs
        self.warmup_steps = self.step_per_epoch * self.warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Current step (not the epoch)
        current_step = self.last_epoch + 1
        
        # Warmup phase
        if current_step <= self.warmup_steps:
            warmup_progress = float(current_step) / float(self.warmup_steps)
            return [self.min_lr + (base_lr - self.min_lr) * warmup_progress for base_lr in self.base_lrs]
        else:
            # Half-cycle cosine decay
            # progress = float(self.last_epoch - self.warmup_epochs) / float(self.total_steps - self.warmup_epochs)
            # scale = 0.5 * (1. + math.cos(math.pi * progress))
            # return [self.min_lr + (base_lr - self.min_lr) * scale for base_lr in self.base_lrs]
            decay_steps = self.total_steps - self.warmup_steps
            progress = float(current_step - self.warmup_steps) / float(decay_steps)
            scale = 0.5 * (1. + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * scale for base_lr in self.base_lrs]