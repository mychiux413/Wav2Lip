from os.path import join
from tqdm import tqdm

from w2l.models import SyncNet_color as SyncNet
from w2l.models import Wav2Lip as Wav2Lip

import torch
from torch import nn
from torch import optim
from torch.utils import data as data_utils
import numpy as np

import os
import cv2
import argparse
from w2l.hparams import hparams
from w2l.utils.data import Wav2LipDataset
from w2l.utils.env import device, use_cuda
from w2l.utils.loss import ms_ssim_loss

global_step = 0
global_epoch = 0

syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False


def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(
        0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(
        0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(
        0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder):
        os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.png'.format(folder, batch_idx, t), c[t])


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


l1loss = nn.L1Loss()


def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(Wav2LipDataset.syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)


def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    # resumed_step = global_step

    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_rec_loss, running_target_loss = 0., 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, indiv_mels, mel, gt) in prog_bar:
            if x.size(0) == 1:
                continue
            model.train()
            optimizer.zero_grad()

            # Move data to CUDA device
            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)

            g, generative_filter = model(indiv_mels, x)
            mouth_g = g * generative_filter
            mouth_gt = gt * generative_filter

            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, g)
            else:
                sync_loss = 0.

            rec_loss = 0.2 * (0.14 * l1loss(g, gt) + ms_ssim_loss(g, gt) * 0.86) + \
                0.8 * (0.14 * l1loss(mouth_g, mouth_gt) + ms_ssim_loss(mouth_g, mouth_gt) * 0.86) / torch.mean(generative_filter)

            loss = hparams.syncnet_wt * sync_loss + \
                (1 - hparams.syncnet_wt) * rec_loss
            loss.backward()
            optimizer.step()

            if global_step % checkpoint_interval == 0:
                save_sample_images(x, g, gt, global_step, checkpoint_dir)

            global_step += 1
            # cur_session_steps = global_step - resumed_step

            running_target_loss += loss.item()
            running_rec_loss += rec_loss.item()
            if hparams.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step == 1 or global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                    average_sync_loss = eval_model(
                        test_data_loader, global_step, device, model, checkpoint_dir)

                    if average_sync_loss < .4:
                        # without image GAN a lesser weight is sufficient
                        hparams.set_hparam('syncnet_wt', 0.01)

            next_step = step + 1
            prog_bar.set_description('Rec: {:04f}, Sync Loss: {:04f}, Target Loss: {:04f}'.format(
                running_rec_loss / next_step,
                running_sync_loss / next_step,
                running_target_loss / next_step,
            ))

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 700
    print('Evaluating for {} steps'.format(eval_steps))
    sync_losses, recon_losses, target_losses = [], [], []
    step = 0
    while 1:
        for x, indiv_mels, mel, gt in test_data_loader:
            step += 1
            if x.size(0) == 1:
                continue
            model.eval()

            # Move data to CUDA device
            x = x.to(device)
            gt = gt.to(device)
            indiv_mels = indiv_mels.to(device)
            mel = mel.to(device)

            g, generative_filter = model(indiv_mels, x)
            mouth_g = g * generative_filter
            mouth_gt = gt * generative_filter

            sync_loss = get_sync_loss(mel, g)
            rec_loss = 0.2 * (0.14 * l1loss(g, gt) + ms_ssim_loss(g, gt) * 0.86) + \
                0.8 * (0.14 * l1loss(mouth_g, mouth_gt) + ms_ssim_loss(mouth_g, mouth_gt) * 0.86) / torch.mean(generative_filter)
            loss = hparams.syncnet_wt * sync_loss + \
                (1 - hparams.syncnet_wt) * rec_loss

            sync_losses.append(sync_loss.item())
            recon_losses.append(rec_loss.item())
            target_losses.append(loss.item())

            if step > eval_steps:
                averaged_sync_loss = np.mean(sync_losses)
                averaged_recon_loss = np.mean(recon_losses)
                averaged_target_loss = np.mean(target_losses)

                print('Rec: {:04f}, Sync loss: {:04f}, Target Loss: {:04f}'.format(
                    averaged_recon_loss, averaged_sync_loss, averaged_target_loss))

                return averaged_sync_loss


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


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model


def main():
    parser = argparse.ArgumentParser(
        description='Code to train the Wav2Lip model without the visual quality discriminator')

    parser.add_argument(
        "--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)

    parser.add_argument(
        '--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
    parser.add_argument('--syncnet_checkpoint_path',
                        help='Load the pre-trained Expert discriminator', required=True, type=str)

    parser.add_argument(
        '--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)
    parser.add_argument('--train_limit', type=int,
                        required=False, default=0)
    parser.add_argument('--val_limit', type=int,
                        required=False, default=0)
    parser.add_argument('--filelists_dir',
                        help='Specify filelists directory', type=str, default='filelists')

    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir

    # Dataset and Dataloader setup
    train_dataset = Wav2LipDataset('train', args.data_root,
                                   sampling_half_window_size_seconds=hparams.sampling_half_window_size_seconds,
                                   unmask_fringe_width=hparams.unmask_fringe_width,
                                   limit=args.train_limit,
                                   filelists_dir=args.filelists_dir)
    test_dataset = Wav2LipDataset('val', args.data_root,
                                  sampling_half_window_size_seconds=hparams.sampling_half_window_size_seconds,
                                  unmask_fringe_width=hparams.unmask_fringe_width, img_augment=False,
                                  limit=args.val_limit,
                                  filelists_dir=args.filelists_dir)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=4)

    # Model
    model = Wav2Lip().to(device)
    print('total trainable params {}'.format(sum(p.numel()
          for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate,
                           amsgrad=hparams.opt_amsgrad, weight_decay=hparams.opt_weight_decay)

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model,
                        optimizer, reset_optimizer=False)

    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None,
                    reset_optimizer=True, overwrite_global_states=False)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    model = nn.DataParallel(model)
    # Train!
    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.checkpoint_interval,
          nepochs=hparams.nepochs)


if __name__ == "__main__":
    main()
