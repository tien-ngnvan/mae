# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    mask_mode=None
                    ):
    # model.mask_mode = mask_mode
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        pixel_values_mask = samples['pixel_values_mask'].to(device, non_blocking=True)
        pixel_values = samples['pixel_values'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
        # with torch.amp.autocast("cuda"):
            loss, _, _ = model(pixel_values, pixel_values_mask, mask_mode=mask_mode)
            # loss, _, _ = self(pixel_values, pixel_values_mask, mask_mode=mask_mode)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged training stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model: torch.nn.Module,
             data_loader: Iterable,
             device: torch.device, epoch: int,
             args=None, log_writer=None,
             mask_mode=None
             ):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 1
    
    accum_iter = args.accum_iter
    
    # Switch to evaluation mode
    model.eval()
    loss_value = []
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        pixel_values_mask = samples['pixel_values_mask'].to(device, non_blocking=True)
        pixel_values = samples['pixel_values'].to(device, non_blocking=True)
        # loss, _, _ = model(pixel_values, pixel_values_mask, mask_ratio=args.mask_ratio)
        loss, _, _ = model(pixel_values, pixel_values_mask, mask_mode=mask_mode)
        loss_value.append(loss.item())

    loss_value = sum(loss_value) / len(loss_value)
    if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        log_writer.add_scalar('val_loss', misc.all_reduce_mean(loss_value), epoch * 1000)
    
    metric_logger.synchronize_between_processes()
    print("Averaged evaluation stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}