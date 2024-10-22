import argparse
import datetime
import json
import numpy as np
import os
import time
import timm
assert timm.__version__ == "0.3.2"  # version check
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm.optim.optim_factory as optim_factory
import util.misc as misc
import timm.models.layers.helpers
import models_mae_update as models_mae

from util.misc import NativeScalerWithGradNormCount as NativeScaler
from pathlib import Path
from util.pos_embed import interpolate_pos_embed
from engine_train import train_one_epoch, evaluate
from data import MAEDataset

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # Training parameters
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--dataset_name_train', type=str, help='dataset name train')
    parser.add_argument('--image_folder', type=str, nargs='+', help='path to image folder')
    parser.add_argument('--do_train', action="store_true", help='do train')
    parser.add_argument('--do_eval', action="store_true", help='do eval')
    parser.add_argument('--max_train_samples', type=int, default=None, help='max train samples')
    parser.add_argument('--max_val_samples', type=int, default=None, help='max val samples')
    parser.add_argument('--num_proc', type=int, default=8, help='num proc')
    parser.add_argument('--streaming', type=bool, default=False, help='streaming')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='mask config')
    parser.add_argument('--mask_min', type=float, default=0, help='mask min')
    parser.add_argument('--mask_max', type=float, default=1, help='mask max')
    parser.add_argument('--cache_dir', type=str, default=".cache", help='cache dir')
    parser.add_argument('--mean_dataset', type=tuple, default=None, help='mean dataset')
    parser.add_argument('--std_dataset', type=tuple, default=None, help='std dataset')

    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Finetune params
    parser.add_argument('--uptrain', default=None, help='load model from checkpoint')
    return parser

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    
    # Fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    
    # Define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    if args.uptrain is not None and os.path.isfile(args.uptrain):
        checkpoint = torch.load(args.uptrain, map_location='cpu')
        
        print("Load pre-trained checkpoint from: %s" % args.uptrain)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    else:
        print("No pre-trained model loaded")
    model.to(device)
    
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # Following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Load the dataset and create dataloader
    dataset = MAEDataset(
        image_folder=args.image_folder,
        dataset_name_train=args.dataset_name_train,
        do_train=args.do_train,
        do_eval=args.do_eval,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        num_proc=args.num_proc,
        streaming=args.streaming,
        batch_size=args.batch_size,
        mask_config=args.mask_ratio,
        img_size=(args.input_size, args.input_size),
        mask_min=args.mask_min,
        mask_max=args.mask_max,
        cache_dir=args.cache_dir,
        mean_dataset=[0.485, 0.456, 0.406],
        std_dataset=[0.229, 0.224, 0.225]
    ).process()
    train_dataset = torch.utils.data.DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True)
    val_dataset = torch.utils.data.DataLoader(dataset['validation'], batch_size=args.batch_size, shuffle=False)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    if args.do_train:
        for epoch in range(args.start_epoch, args.epochs):
            train_stats = train_one_epoch(
                model, train_dataset,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )
            if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
                    
            if args.do_eval:
               test_stats = evaluate(model, val_dataset, device, args) 
               log_stats = {**{f'train_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,}

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
        
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
