import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

from llama import LLaMA_adapter

def train_one_epoch(model: LLaMA_adapter,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    # model.module.set_default_trainability()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # hank
        examples = batch["input_ids"]
        labels = batch["labels"]
        example_mask = batch["input_mask"]
        imgs = batch["audio"]

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
             c_loss, m_loss = model(examples, labels, imgs)
        loss = c_loss  + m_loss * 0
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        m_loss_value = m_loss
        # if not math.isfinite(loss_value):
        #     print("Loss is {}({}), stopping training".format(loss_value, loss))
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            # sys.exit(1)

        loss /= accum_iter
        norm = loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=5.0,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if not math.isfinite(loss_value):
            print("Loss is {}({}), norm={}".format(loss_value, loss, norm))


        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        metric_logger.update(mloss=m_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        m_loss_value_reduce = misc.all_reduce_mean(m_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('m_train_loss', m_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(model: LLaMA_adapter, data_loader: Iterable, device: torch.device, epoch=0):
    model.eval()
    val_loss = 0
    for data_iter_step, batch in enumerate(data_loader):
        # hank
        examples = batch["input_ids"]
        labels = batch["labels"]
        example_mask = batch["input_mask"]
        imgs = batch["audio"]

        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
             c_loss, m_loss = model(examples, labels, imgs)
        loss = c_loss  + m_loss * 0
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        m_loss_value = m_loss

        val_loss += loss_value

    print(f"Validation loss: {val_loss / len(data_loader)}")
    return {
        "epoch": epoch, 
        "val_loss": val_loss / len(data_loader)
    }