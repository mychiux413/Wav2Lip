from os.path import join
from tqdm import tqdm

from w2l.models import SyncNet_color as SyncNet

import torch
from torch import nn
from torch import optim
from torch.utils import data as data_utils


import os
import argparse
from w2l.hparams import hparams
from w2l.utils.data import SyncnetDataset
from w2l.utils.env import use_cuda, device

global_step = 0
global_epoch = 0

logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    # resumed_step = global_step

    while global_epoch < nepochs:
        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel, y) in prog_bar:
            if x.size(0) == 1:
                continue

            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            # cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step,
                               device, model, checkpoint_dir)

            prog_bar.set_description(
                'Loss: {}'.format(running_loss / (step + 1)))

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):
            if x.size(0) == 1:
                continue

            model.eval()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            losses.append(loss.item())

            if step > eval_steps:
                break

        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)

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
    global_epoch = checkpoint["global_epoch"]

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
        parser.add_argument('--train_limit', type=int, required=False, default=0)
        parser.add_argument('--val_limit', type=int, required=False, default=0)
        parser.add_argument('--filelists_dir',
                            help='Specify filelists directory', type=str, default='filelists')

        args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup
    train_dataset = SyncnetDataset('train', args.data_root, limit=args.train_limit,
                                   sampling_half_window_size_seconds=1e10,
                                   filelists_dir=args.filelists_dir)
    val_dataset = SyncnetDataset('val', args.data_root, limit=args.val_limit,
                                 sampling_half_window_size_seconds=1e10,
                                 img_augment=False,
                                 filelists_dir=args.filelists_dir)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=hparams.num_workers)

    val_data_loader = data_utils.DataLoader(
        val_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=8)

    # Model
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
                        optimizer, reset_optimizer=False)

    model = nn.DataParallel(model)
    train(device, model, train_data_loader, val_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)


if __name__ == "__main__":
    main()
