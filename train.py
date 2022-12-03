# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import time

from datetime import timedelta

import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.modeling import VisionTransformer, CONFIGS, LabelSmoothing
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule, WarmupConstantSchedule
from utils.data_utils import get_loader
from utils.autoaugment import mixup_criterion, mixup_data
from torch.nn import CrossEntropyLoss
from utils.loss_util import con_loss, PairwiseLoss, FocalLoss
import math

logger = logging.getLogger(__name__)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    rt /= nprocs
    return rt


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_model(args, model, optimizer, lr_schedule, epoch):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "{}_checkpoint.bin".format(args.name))
    checkpoint = {
        'model': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_schedule': lr_schedule.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089
    elif args.dataset == "blood_cell_23":
        num_classes = 23
    elif args.dataset == "blood_cell_40":
        num_classes = 40
    elif args.dataset == "PBC":
        num_classes = 8
    elif args.dataset == "ImageNet-1K":
        num_classes = 1000

    print(num_classes)
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    # model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, epoch):
    # Validation!
    eval_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    n_iter_test = len(test_loader)
    loss_fct = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            _, logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())
            prec1, prec5 = accuracy(logits, y, topk=(1, 5))
            top1.update(prec1[0], x.size(0))
            top5.update(prec5[0], x.size(0))

            epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
            writer.add_scalar("test/eval_loss", scalar_value=eval_losses.avg,
                              global_step=(epoch / 10 - 1) * n_iter_test + step)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global epochs: %d" % epoch)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid top-1 Accuracy: %2.5f" % top1.avg)
    logger.info("valid top-5 Accuracy %2.5f" % top5.avg)
    if args.local_rank in [-1, 0]:
        writer.add_scalar("test/top1-accuracy", scalar_value=top1.avg, global_step=epoch)
        writer.add_scalar("test/top5-accuracy", scalar_value=top5.avg, global_step=epoch)

    return top1.avg, top5.avg


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)
    # train epochs
    epochs = args.epochs
    # Prepare optimizer and scheduler
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.learning_rate,
                                      betas=(0.9, 0.999), eps=1e-08,
                                      weight_decay=args.weight_decay)

    # Prepare the lr scheduler
    iter_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - args.warmup_epochs)
    warmup_scheduler = WarmUpLR(optimizer, args.warmup_epochs * iter_per_epoch)

    start_epoch = -1
    # loss function
    if args.loss == "CE":
        criterion = CrossEntropyLoss()
    elif args.loss == "Focalloss":
        criterion = FocalLoss(40, alpha=args.alp, gamma=2, size_average=True)
    elif args.loss == "Pairwiseloss":
        criterion = PairwiseLoss(23)
    elif args.loss == "LabelSmoothing":
        criterion = LabelSmoothing(args.smoothing_value)

    global_step, best_acc = 0, 0

    if args.pretrained_model is not None:
        print("Pretrained Model")
        checkpoints_path = torch.load(args.pretrained_model)
        pretrained_model = checkpoints_path['model']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        #optimizer.load_state_dict(checkpoints_path['optimizer'])
        #start_epoch = checkpoints_path['epoch']
        #scheduler.load_state_dict(checkpoints_path['lr_schedule'])
        #global_step = (start_epoch + 1) * len(train_loader)

    model.to(args.device)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    start_time = time.time()
    for epoch in range(start_epoch + 1, epochs + 1):
        logger.info("Train at epoch {}".format(epoch))

        model.train()
        epoch_start_time = time.time()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X Epochs / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs, targets = batch
            # n_iter = (epoch - 1)*len(train_loader) + step + 1
            # using mixup
            if args.alpha > 0:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, use_cuda=True)

                part_token, logits = model(inputs)
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                loss = loss_func(criterion, logits)

                prec1_1, prec5_1 = accuracy(logits, targets_a, topk=(1, 5))
                prec1_2, prec5_2 = accuracy(logits, targets_b, topk=(1, 5))

                prec1 = lam * prec1_1[0] + (1 - lam) * prec1_2[0]
                prec5 = lam * prec5_1[0] + (1 - lam) * prec5_2[0]

            else:
                part_token, logits = model(inputs)
                loss = criterion(logits, targets)
                prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
                prec1, prec5 = prec1[0], prec5[0]

            contrast_loss = con_loss(part_token, targets)
            loss += contrast_loss
            loss = loss.mean()

            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))
            losses.update(loss.item() * args.gradient_accumulation_steps)

            if epoch <= args.warmup_epochs:
                warmup_scheduler.step()
            else:
                scheduler.step(epoch + step / iter_per_epoch)
            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            global_step += 1
            epoch_iterator.set_description(
                "Training (%d Epochs / %d Steps) (loss=%2.5f)" % (epoch, global_step, losses.val)
            )

            writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=optimizer.param_groups[-1]['lr'], global_step=global_step)

        epoch_end_time = time.time()
        logger.info("Train epoch time {}".format((epoch_end_time - epoch_start_time) / 3600))
        logger.info("train top1 accuracy so far: %f" % top1.avg)
        logger.info("train top5 accuracy so far: %f" % top5.avg)
        writer.add_scalar("train/train_top1-accuracy", scalar_value=top1.avg, global_step=epoch)
        writer.add_scalar("train/train_top5-accuracy", scalar_value=top5.avg, global_step=epoch)

        if epoch % args.eval_every == 0:
            top1_acc, top5_acc = valid(args, model, writer, test_loader, epoch)
            if best_acc < top1_acc:
                save_model(args, model, optimizer, scheduler, epoch)
                best_acc = top1_acc
            logger.info("best top1 accuracy so far: %f" % best_acc)

    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", default="blood_cell_23", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset",
                        choices=["blood_cell_23", "blood_cell_40", "CUB_200_2011", "car", "dog", "nabirds", "INat2017",
                                 "PBC", "ImageNet-1K"], default="blood_cell_23",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='/home/vipuser/Mywork')
    parser.add_argument("--model_type", choices=["ViT-lite_16", "ViT-S_16, ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="/home/vipuser/Mywork/cb/CNNStem_Local/output/CNNStem_Local_AdamW_ImageNet-1k_mixup_checkpoint.bin",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default="/home/vipuser/Mywork/cb/CNNStem_Local/output/CNNStem_Local_AdamW_ImageNet-1k_mixup_checkpoint.bin",
                        help="load pretrained model")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=10, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--learning_rate", default=0.03, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=1e-5, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--warmup_epochs", default=5, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--smoothing_value', type=float, default=0.1,
                        help="Label smoothing value\n")

    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    parser.add_argument('--imbalanced', type=bool, default=True, help="balance the dataset by select data")
    parser.add_argument('--alpha', type=float, default=0.8, help="the paramater to mixup data")
    parser.add_argument('--alp', type=int, nargs=40, help="the weight of different type")
    parser.add_argument('--loss', choices=["CE", "Focalloss", "Pairwiseloss", "LabelSmoothing"],
                        default="LabelSmoothing")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optimizer', choices=["SGD", "AdamW"], default="SGD")

    args = parser.parse_args()

    # if args.fp16 and args.smoothing_value != 0:
    #     raise NotImplementedError("label smoothing not supported for fp16 training now")
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1

    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))

    args.n_gpu = 1
    args.device = device
    args.nprocs = 1

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)
    # Training
    train(args, model)


if __name__ == "__main__":
    main()
