from os.path import join
from tqdm import tqdm

from w2l.models import SyncNet_color as SyncNet
from w2l.models import SyncNet_shuffle_color

import torch
from torch import nn
from torch import optim
from torch.utils import data as data_utils


import os
import argparse
from w2l.hparams import hparams
from w2l.utils.data import SyncnetDataset
from w2l.utils.env import use_cuda, device
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random
from time import time
from datetime import datetime

global_step = 0
global_epoch = 0

logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None, K=1,
          summary_writer=None):

    model.train()
    global global_step, global_epoch
    # resumed_step = global_step

    if hparams.warm_up_epochs > 0:
        C = np.log(hparams.syncnet_lr /
                   hparams.syncnet_min_lr) / hparams.warm_up_epochs

    half_img_size = hparams.img_size // 2
    while global_epoch < nepochs:
        if global_epoch < hparams.warm_up_epochs:
            lr = hparams.syncnet_min_lr * np.exp(C * global_epoch)
        else:
            lr = hparams.syncnet_lr * \
                (hparams.syncnet_lr_decay_rate **
                 (global_epoch - hparams.warm_up_epochs))
            lr = max(hparams.syncnet_min_lr, lr)
        print("epoch: {}, lr: {}".format(global_epoch, lr))
        if summary_writer is not None:
            summary_writer.add_scalar("LearningRate/Train", lr, global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        running_loss, running_fake_loss, running_real_loss = 0., 0., 0.
        
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel) in prog_bar:
            # x: B x 2T x 3 x H x W
            B = x.size(0)
            if B == 1:
                continue

            # x_true B x T x 3 x H x W
            x_true = x[:, :hparams.syncnet_T, :]

            # x_false: B x T x 3 x H x W
            x_false = x[:, hparams.syncnet_T:, :]
            x_true = x_true.reshape(
                (B, hparams.syncnet_T * 3, half_img_size, hparams.img_size))
            x_false = x_false.reshape(
                (B, hparams.syncnet_T * 3, half_img_size, hparams.img_size))

            y_true = torch.ones((B, 1), dtype=torch.float32, device=device)
            y_false = torch.zeros((B, 1), dtype=torch.float32, device=device)

            # Transform data to CUDA device
            x_true = x_true.to(device)
            x_false = x_false.to(device)

            mel = mel.to(device)

            a, v = model(mel, x_true)
            loss_true = cosine_loss(a, v, y_true)
            a, v = model(mel, x_false)
            loss_false = cosine_loss(a, v, y_false)
            # y = y.to(device)

            loss = (loss_true * 0.625 + loss_false * 0.375) / K
            loss.backward()

            if global_step % K == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            # cur_session_steps = global_step - resumed_step
            running_loss += loss.detach()
            running_fake_loss += loss_false.detach()
            running_real_loss += loss_true.detach()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step,
                               device, model, checkpoint_dir,
                               summary_writer=summary_writer)
                model.train()

            if global_step % K == 0:
                next_step = step + 1

                _loss = running_loss.item() * K / next_step
                _real = running_real_loss.item() / next_step
                _fake = running_fake_loss.item() / next_step
                prog_bar.set_description('Fake: {}, Real: {}, Loss: {}'.format(_fake, _real, _loss))
                if summary_writer is not None:
                    summary_writer.add_scalar(
                        'Train/Loss', _loss, global_step)
                    summary_writer.add_scalar(
                        'Train/Fake', _fake, global_step)
                    summary_writer.add_scalar(
                        'Train/Real', _real, global_step)

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir,
               summary_writer=None):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses, real_losses, fake_losses = [], [], []
    half_img_size = hparams.img_size // 2
    while 1:
        for step, (x, mel) in enumerate(test_data_loader):
            B = x.size(0)
            if B == 1:
                continue

            # x_true B x T x 3 x H x W
            x_true = x[:, :hparams.syncnet_T, :]
            x_false = x[:, hparams.syncnet_T:, :]
            x_true = x_true.reshape(
                (B, hparams.syncnet_T * 3, half_img_size, hparams.img_size))
            x_false = x_false.reshape(
                (B, hparams.syncnet_T * 3, half_img_size, hparams.img_size))
            y_true = torch.ones((B, 1), dtype=torch.float32, device=device)
            y_false = torch.zeros((B, 1), dtype=torch.float32, device=device)
            # x = torch.cat([
            #     x_true,
            #     x_false,
            # ], dim=0)

            model.eval()

            # Transform data to CUDA device
            # x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x_true)
            loss_true = cosine_loss(a, v, y_true)
            a, v = model(mel, x_false)
            loss_false = cosine_loss(a, v, y_false)

            # a, v = model(mel, x)
            # y = y.to(device)

            # loss = cosine_loss(a, v, y)
            _t = loss_true.detach()
            _f = loss_false.detach()
            losses.append((_t + _f) / 2.)
            real_losses.append(_t)
            fake_losses.append(_f)

            if step > eval_steps:
                break

        _losses = [loss.item() for loss in losses]
        _real_losses = [loss.item() for loss in real_losses]
        _fake_losses = [loss.item() for loss in fake_losses]
        _loss = np.mean(_losses)
        _real = np.mean(_real_losses)
        _fake = np.mean(_fake_losses)
        print('Evaluation | Fake: {}, Real: {}, Loss: {}'.format(_fake, _real, _loss))
        if summary_writer is not None:
            summary_writer.add_scalar(
                'Evaluation/Loss', _loss, global_step)
            summary_writer.add_scalar(
                'Evaluation/Real', _real, global_step)
            summary_writer.add_scalar(
                'Evaluation/Fake', _fake, global_step)

        return


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.module.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"] + 1
    if reset_optimizer:
        global_step = 0
        global_epoch = 0

    return model


def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser(
            description='Code to train the expert lip-sync discriminator')

        parser.add_argument(
            "--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)

        parser.add_argument('--checkpoint_dir',
                            help='Save checkpoints to this directory', required=True, type=str)
        parser.add_argument('--checkpoint_path',
                            help='Resumed from this checkpoint', default=None, type=str)
        parser.add_argument('--reset_optimizer',
                            help='Reset optimizer or not', action='store_true')
        parser.add_argument('--train_limit', type=int,
                            required=False, default=0)
        parser.add_argument('--val_limit', type=int, required=False, default=0)
        parser.add_argument('--filelists_dir',
                            help='Specify filelists directory', type=str, default='filelists')
        parser.add_argument('--hparams',
                            help='specify hparams file, default is None, this overwrite is after the env overwrite',
                            type=str, default=None)
        parser.add_argument('--K',
                            help='Delay update', type=int, default=1)
        parser.add_argument('--shufflenet',
                            help='Use Shuffle net as faceencoder', action='store_true')

        args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    now = datetime.now()
    log_dir = os.path.join(
        checkpoint_dir, "log-syncnet-{}".format(now.strftime("%Y-%m-%d-%H_%M")))
    print("log at: {}".format(log_dir))
    summary_writer = SummaryWriter(log_dir)

    if args.hparams is None:
        dump_hparams_path = os.path.join(
            checkpoint_dir, "hparams-syncnet-{}.json".format(now.strftime("%Y-%m-%d-%H_%M")))
        hparams.to_json(dump_hparams_path)
    else:
        hparams.overwrite_by_json(args.hparams)

    # Dataset and Dataloader setup
    train_dataset = SyncnetDataset(
        'train', args.data_root, limit=args.train_limit,
        img_augment=hparams.img_augment,
        sampling_half_window_size_seconds=1e10,
        filelists_dir=args.filelists_dir,
        inner_shuffle=False)
    val_dataset = SyncnetDataset(
        'val', args.data_root, limit=args.val_limit,
        sampling_half_window_size_seconds=1e10,
        img_augment=False,
        filelists_dir=args.filelists_dir,
        inner_shuffle=False)

    def worker_init_fn(i):
        seed = int(time()) + i * 100
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        return

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=hparams.num_workers,
        pin_memory=use_cuda,
        worker_init_fn=worker_init_fn,
        shuffle=True)

    val_data_loader = data_utils.DataLoader(
        val_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=max(1, hparams.num_workers // 2),
        worker_init_fn=worker_init_fn)

    # Model
    if args.shufflenet:
        print("**** Enable ShuffleNet V2 1.0 as syncnet ****")
        model = SyncNet_shuffle_color().to(device)
    else:
        model = SyncNet().to(device)

    print('total trainable params {}'.format(sum(p.numel()
          for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=hparams.syncnet_lr,
        amsgrad=hparams.syncnet_opt_amsgrad,
        weight_decay=hparams.syncnet_opt_weight_decay,
    )

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model,
                        optimizer, reset_optimizer=args.reset_optimizer)

    model = nn.DataParallel(model)
    train(device, model, train_data_loader, val_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs, K=args.K,
          summary_writer=summary_writer)


if __name__ == "__main__":
    main()
