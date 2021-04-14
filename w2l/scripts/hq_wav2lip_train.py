from os.path import join
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils import data as data_utils
import numpy as np
import os
import cv2
import argparse
from w2l.hparams import hparams
from w2l.utils.data import Wav2LipDataset
from w2l.utils.env import use_cuda, device
from w2l.models import SyncNet_color as SyncNet
from w2l.models import Wav2Lip, Wav2Lip_disc_qual
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


def get_landmarks_loss(g_landmarks, gt_landmarks):
    axis_delta = g_landmarks[:, :, :, 0] - gt_landmarks[:, :, :, 0] + g_landmarks[:, :, :, 1] - gt_landmarks[:, :, :, 1]
    square_mean = torch.mean(torch.mean(axis_delta ** 2, dim=0), dim=-1)
    loss_sum_T = torch.sum(square_mean)
    return loss_sum_T / hparams.syncnet_T


def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(hparams.syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)


def train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    # resumed_step = global_step
    landmarks_points_len = len(hparams.landmarks_points)

    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_rec_loss, running_perceptual_loss = 0., 0., 0.
        running_disc_real_loss, running_disc_fake_loss, running_target_loss = 0., 0., 0.
        running_landmarks_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, indiv_mels, mel, gt, landmarks) in prog_bar:
            B = x.size(0)
            if B == 1:
                continue
            disc.train()
            model.train()

            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)

            # Train generator now. Remove ALL grads.
            optimizer.zero_grad()
            disc_optimizer.zero_grad()

            g, g_landmarks = model(indiv_mels, x)

            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, g)
            else:
                sync_loss = 0.

            if hparams.disc_wt > 0.:
                perceptual_loss = disc.perceptual_forward(g)
            else:
                perceptual_loss = 0.

            gt_landmarks = landmarks[:, :, hparams.landmarks_points].reshape(
                (B, hparams.syncnet_T, landmarks_points_len, 2))
            g_landmarks = g_landmarks.reshape((B, hparams.syncnet_T, landmarks_points_len, 2))
            gt_landmarks = gt_landmarks.to(device)
            landmarks_loss = get_landmarks_loss(g_landmarks, gt_landmarks)

            rec_loss = hparams.l1_wt * \
                l1loss(g, gt) + (1.0 - hparams.l1_wt) * ms_ssim_loss(g, gt)

            loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
                (1. - hparams.syncnet_wt - hparams.disc_wt) * \
                rec_loss + landmarks_loss * hparams.landmarks_wt

            loss.backward()
            optimizer.step()

            # Remove all gradients before Training disc
            disc_optimizer.zero_grad()

            pred = disc(gt)
            disc_real_loss = F.binary_cross_entropy(
                pred, torch.ones((len(pred), 1)).to(device))
            disc_real_loss.backward()

            pred = disc(g.detach())
            disc_fake_loss = F.binary_cross_entropy(
                pred, torch.zeros((len(pred), 1)).to(device))
            disc_fake_loss.backward()

            disc_optimizer.step()

            running_disc_real_loss += disc_real_loss.item()
            running_disc_fake_loss += disc_fake_loss.item()
            running_landmarks_loss += landmarks_loss.item()

            if global_step % checkpoint_interval == 0:
                save_sample_images(x, g, gt, global_step, checkpoint_dir)

            # Logs
            global_step += 1
            # cur_session_steps = global_step - resumed_step

            running_target_loss += loss.item()
            running_rec_loss += rec_loss.item()
            if hparams.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.

            if hparams.disc_wt > 0.:
                running_perceptual_loss += perceptual_loss.item()
            else:
                running_perceptual_loss += 0.

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)
                save_checkpoint(disc, disc_optimizer, global_step,
                                checkpoint_dir, global_epoch, prefix='disc_')

            if global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                    average_sync_loss = eval_model(
                        test_data_loader, global_step, device, model, disc)

                    if average_sync_loss < .55:
                        hparams.set_hparam('syncnet_wt', 0.03)

            next_step = step + 1
            prog_bar.set_description(
                'Rec: {:04f}, Sync: {:04f}, Percep: {:04f} | Fake: {:04f}, Real: {:04f} | Landmarks: {:04f} | Target: {:04f}'.format(
                    running_rec_loss / next_step,
                    running_sync_loss / next_step,
                    running_perceptual_loss / next_step,
                    running_disc_fake_loss / next_step,
                    running_disc_real_loss / next_step,
                    running_landmarks_loss / next_step,
                    running_target_loss / next_step,
                ))

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, disc):
    eval_steps = 300
    print('Evaluating for {} steps'.format(eval_steps))
    running_sync_loss, recon_losses, running_disc_real_loss, \
        running_disc_fake_loss, running_perceptual_loss, running_target_loss, \
        running_landmarks_loss = [], [], [], [], [], [], []

    landmarks_points_len = len(hparams.landmarks_points)
    for step, (x, indiv_mels, mel, gt, landmarks) in enumerate((test_data_loader)):
        B = x.size(0)
        if B == 1:
            continue
        model.eval()
        disc.eval()

        x = x.to(device)
        mel = mel.to(device)
        indiv_mels = indiv_mels.to(device)
        gt = gt.to(device)

        pred = disc(gt)
        disc_real_loss = F.binary_cross_entropy(
            pred, torch.ones((len(pred), 1)).to(device))

        g, g_landmarks = model(indiv_mels, x)

        gt_landmarks = landmarks[:, :, hparams.landmarks_points].reshape(
            (B, hparams.syncnet_T, landmarks_points_len, 2))
        g_landmarks = g_landmarks.reshape((B, hparams.syncnet_T, landmarks_points_len, 2))
        gt_landmarks = gt_landmarks.to(device)
        landmarks_loss = get_landmarks_loss(g_landmarks, gt_landmarks)

        pred = disc(g)
        disc_fake_loss = F.binary_cross_entropy(
            pred, torch.zeros((len(pred), 1)).to(device))

        running_disc_real_loss.append(disc_real_loss.item())
        running_disc_fake_loss.append(disc_fake_loss.item())
        running_landmarks_loss.append(landmarks_loss.item())

        sync_loss = get_sync_loss(mel, g)

        if hparams.disc_wt > 0.:
            perceptual_loss = disc.perceptual_forward(g)
        else:
            perceptual_loss = 0.

        rec_loss = hparams.l1_wt * l1loss(g, gt) + (1.0 - hparams.l1_wt) * ms_ssim_loss(g, gt)

        loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
            (1. - hparams.syncnet_wt - hparams.disc_wt) * \
            rec_loss + hparams.landmarks_wt * landmarks_loss

        running_target_loss.append(loss.item())
        recon_losses.append(rec_loss.item())
        running_sync_loss.append(sync_loss.item())

        if hparams.disc_wt > 0.:
            running_perceptual_loss.append(perceptual_loss.item())
        else:
            running_perceptual_loss.append(0.)

        if step > eval_steps:
            break

    print('Rec: {:04f}, Sync: {:04f}, Percep: {:04f} | Fake: {:04f}, Real: {:04f} | Landmarks: {:04f} | Target: {:04f}'.format(
        np.mean(recon_losses),
        np.mean(running_sync_loss),
        np.mean(running_perceptual_loss),
        np.mean(running_disc_fake_loss),
        np.mean(running_disc_real_loss),
        np.mean(running_landmarks_loss),
        np.mean(running_target_loss)))
    return np.mean(running_sync_loss)


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict() if getattr(model, 'module', None) is None else model.module.state_dict(),
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


def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser(
            description='Code to train the Wav2Lip model WITH the visual quality discriminator')

        parser.add_argument(
            "--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)

        parser.add_argument(
            '--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
        parser.add_argument('--syncnet_checkpoint_path',
                            help='Load the pre-trained Expert discriminator', required=True, type=str)

        parser.add_argument(
            '--checkpoint_path', help='Resume generator from this checkpoint', default=None, type=str)
        parser.add_argument('--disc_checkpoint_path',
                            help='Resume quality disc from this checkpoint', default=None, type=str)
        parser.add_argument('--train_limit', type=int,
                            required=False, default=0)
        parser.add_argument('--val_limit', type=int, required=False, default=0)
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
    disc = Wav2Lip_disc_qual().to(device)

    print('total trainable params {}'.format(sum(p.numel()
          for p in model.parameters() if p.requires_grad)))
    print('total DISC trainable params {}'.format(sum(p.numel()
          for p in disc.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate, betas=(
        0.5, 0.999),
        amsgrad=hparams.opt_amsgrad, weight_decay=hparams.opt_weight_decay)
    disc_optimizer = optim.Adam([p for p in disc.parameters() if p.requires_grad],
                                lr=hparams.disc_initial_learning_rate, betas=(
        0.5, 0.999),
        amsgrad=hparams.disc_opt_amsgrad, weight_decay=hparams.disc_opt_weight_decay)

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model,
                        optimizer, reset_optimizer=False)

    if args.disc_checkpoint_path is not None:
        load_checkpoint(args.disc_checkpoint_path, disc, disc_optimizer,
                        reset_optimizer=False, overwrite_global_states=False)

    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True,
                    overwrite_global_states=False)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    model = nn.DataParallel(model)
    # Train!
    train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.checkpoint_interval,
          nepochs=hparams.nepochs)


if __name__ == "__main__":
    main()
