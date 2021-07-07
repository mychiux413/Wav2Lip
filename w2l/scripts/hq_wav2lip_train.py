from os.path import join
from posix import XATTR_CREATE
import torchvision
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
from w2l.models import SyncNet_shuffle_color
from w2l.models import Wav2Lip, Wav2Lip_disc_qual, InceptionV3_disc
from w2l.utils.loss import ms_ssim_loss, cal_blur
from torch.utils.tensorboard import SummaryWriter
import random
from time import time
from datetime import datetime

global_step = 0
global_epoch = 0


def reset_global():
    global global_step
    global global_epoch
    global_step = 0
    global_epoch = 0

    SEED = 4321
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def debug_dump(tensor, to_path='temp/temp.png'):
    imgs = (tensor.detach().cpu().numpy().transpose(
        0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    dirname = os.path.dirname(to_path)
    basename = os.path.basename(to_path)
    lname, ext = os.path.splitext(basename)
    for batch_idx, c in enumerate(imgs):
        for t in range(len(c)):
            cv2.imwrite('{}/{}-{}_{}.{}'.format(dirname,
                        lname, batch_idx, t, ext), c[t])


def save_sample_images(x, g, gt, global_step, checkpoint_dir, g_landmarks=None, gt_landmarks=None, summary_writer=None):
    # g: (B, T, 3, H, W)
    # g_landmarks: (B, T, landmarks_points_len, 2)
    x = (x.detach().cpu().numpy().transpose(
        0, 1, 3, 4, 2) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(
        0, 1, 3, 4, 2) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(
        0, 1, 3, 4, 2) * 255.).astype(np.uint8)
    if g_landmarks is not None and gt_landmarks is not None:
        g_landmarks = g_landmarks.detach().cpu().numpy()
        gt_landmarks = gt_landmarks.detach().cpu().numpy()
    # gt: (B, T, H, W, 3)
    refs, inps = x[..., 3:], x[..., :3]

    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder):
        os.mkdir(folder)

    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, (c, l_g, l_gt) in enumerate(zip(collage, g_landmarks, gt_landmarks)):
        marked_images = []
        for t in range(len(c)):
            img = c[t]
            landmarks_g = l_g[t]
            landmarks_gt = l_gt[t]
            for ratio_x, ratio_y in landmarks_g:
                pos_x = int(hparams.img_size * 3 + hparams.img_size * ratio_x)
                pos_y = int(hparams.img_size * ratio_y)
                img[(pos_y - 1):(pos_y + 1), (pos_x - 1):(pos_x + 1), 1] = 255
            for ratio_x, ratio_y in landmarks_gt:
                pos_x = int(hparams.img_size * 3 + hparams.img_size * ratio_x)
                pos_y = int(hparams.img_size * ratio_y)
                img[(pos_y - 2):(pos_y + 2), (pos_x - 2):(pos_x + 2), 2] = 255
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])
            marked_images.append(img)
        if summary_writer is not None:
            marked_images = np.concatenate(marked_images, axis=0)[:, :, ::-1].astype(np.float) / 255.0
            marked_images = np.expand_dims(marked_images, 0)
            marked_images = torch.from_numpy(marked_images).transpose(1, 3).transpose(2, 3)
            grid = torchvision.utils.make_grid(marked_images)
            summary_writer.add_image('samples_step{:09d}/{:02d}'.format(global_step, batch_idx), grid, 0)


logloss = nn.BCELoss(reduction='none')


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


l1loss = nn.L1Loss(reduction='none')


def get_landmarks_loss(g_landmarks, gt_landmarks):
    # g_landmarks: # (B, T, 14, 2)
    axis_delta = g_landmarks[:, :, :, 0] - gt_landmarks[:, :, :, 0] + \
        g_landmarks[:, :, :, 1] - gt_landmarks[:, :, :, 1]
    square_sum = torch.sum(axis_delta ** 2, dim=2)
    loss_mean_T = torch.mean(square_sum, dim=1)
    return loss_mean_T


def get_sync_loss(syncnet, mel, half_g, expect_true=True):
    B = half_g.size(0)
    # half_g: B x T x 3 x H//2 x W
    half_g = half_g.reshape((B, hparams.syncnet_T * 3, hparams.half_img_size, hparams.img_size))
    # B, T * 3, H//2, W
    a, v = syncnet(mel, half_g)
    if expect_true:
        y = torch.ones((B, 1), dtype=torch.float32, device=device)
    else:
        y = torch.zeros((B, 1), dtype=torch.float32, device=device)
    return cosine_loss(a, v, y).reshape((B,))


def get_blurs_loss(half_g, blurs_gt, B):

    half_g_for_cv2 = (half_g.permute((0, 1, 3, 4, 2)).reshape(
        (B * hparams.syncnet_T, hparams.half_img_size, hparams.img_size, 3)) * 255.).detach().cpu().numpy().astype(np.uint8)
    blurs_g = list(map(cal_blur, half_g_for_cv2))

    blurs_gt = blurs_gt.mean(1)
    blurs_g = torch.FloatTensor(np.array(blurs_g, np.float)).to(device).reshape((B, hparams.syncnet_T)).mean(1)

    return torch.maximum(torch.zeros((B,), device=device, dtype=torch.float32), blurs_g - blurs_gt)


l2loss = nn.MSELoss(reduction='none')


def train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
          syncnet, checkpoint_dir=None, checkpoint_interval=None, nepochs=None, K=1,
          summary_writer=None):
    global global_step, global_epoch

    original_disc_wt = hparams.disc_wt
    origin_syncnet_wt = hparams.syncnet_wt
    init_syncnet_wt = origin_syncnet_wt / 4.0
    hparams.set_hparam('syncnet_wt', init_syncnet_wt)

    init_disc_wt = original_disc_wt / 4.0
    hparams.set_hparam('disc_wt', init_disc_wt)
    half_img_size = hparams.img_size // 2

    if hparams.warm_up_epochs > 0:
        C = np.log(hparams.initial_learning_rate /
                   hparams.min_learning_rate) / hparams.warm_up_epochs
        C_disc = np.log(hparams.disc_initial_learning_rate /
                        hparams.disc_min_learning_rate) / hparams.warm_up_epochs

    print("WarmUp Epochs:", hparams.warm_up_epochs)
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        if global_epoch < hparams.warm_up_epochs:
            lr = hparams.min_learning_rate * np.exp(C * global_epoch)
            disc_lr = hparams.disc_min_learning_rate * \
                np.exp(C_disc * global_epoch)
        else:
            lr = hparams.initial_learning_rate * \
                (hparams.learning_rate_decay_rate **
                 (global_epoch - hparams.warm_up_epochs))
            lr = max(hparams.min_learning_rate, lr)

            disc_lr = hparams.disc_initial_learning_rate * \
                (hparams.disc_learning_rate_decay_rate **
                 (global_epoch - hparams.warm_up_epochs))
            disc_lr = max(hparams.disc_min_learning_rate, disc_lr)

        print("epoch: {}, lr: {}, disc_lr: {}".format(
            global_epoch, lr, disc_lr))
        if summary_writer is not None:
            summary_writer.add_scalar("Train/Wav2Lip-LR", lr, global_step)
            summary_writer.add_scalar("Train/Disc-LR", disc_lr, global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in disc_optimizer.param_groups:
            param_group['lr'] = disc_lr

        running_sync_loss, running_perceptual_loss = 0., 0.
        running_disc_real_loss, running_disc_fake_loss, running_target_loss = 0., 0., 0.
        running_l1_loss, running_ssim_loss, running_disc_loss = 0., 0., 0.
        running_sync_real_loss = 0.
        running_landmarks_loss = 0.
        running_blurs_loss = 0.

        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, indiv_mels, mel, gt, landmarks_gt, blurs_gt, weights, masks) in prog_bar:
            B = x.size(0)
            if B == 1:
                continue

            disc.train()
            model.train()
            syncnet.eval()

            x = x.to(device)
            half_x = x[:, :, :, hparams.half_img_size:]
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)
            landmarks_gt = landmarks_gt.to(device)
            weights = weights.to(device)
            blurs_gt = blurs_gt.to(device)
            masks = masks.to(device)
            bypass_mouth = 1.0 - masks
            half_bypass_mouth = bypass_mouth[:, :, :, half_img_size:]
            half_g, landmarks_g = model(indiv_mels, x)
            upper_gt = gt[:, :, :, :half_img_size]
            g = torch.cat([upper_gt, half_g], dim=3)
            # g: (B, T, 3, img_size, img_size)

            landmarks_loss = get_landmarks_loss(landmarks_g, landmarks_gt)

            half_gt = gt[:, :, :, half_img_size:]

            sync_real_loss = get_sync_loss(
                syncnet, mel, half_gt, expect_true=True) * weights

            blurs_loss = get_blurs_loss(half_g, blurs_gt, B)

            sync_loss = get_sync_loss(syncnet, mel, half_g)

            sync_loss = torch.maximum(sync_loss * weights - sync_real_loss, torch.zeros((B,), device=device, dtype=torch.float32))

            perceptual_loss = disc.perceptual_forward(half_g, half_x)
            perceptual_loss = perceptual_loss.reshape((B, hparams.syncnet_T)).mean(1)

            # masks_for_g = masks.permute((0, 2, 1, 3, 4))

            # debug_dump(ssim_gt, 'temp/ssim_gt.png')
            # debug_dump(ssim_g, 'temp/ssim_g.png')
            # refs, inps = x[:, 3:], x[:, :3]
            # debug_dump(refs, 'temp/refs.png')
            # debug_dump(inps, 'temp/inps.png')

            # masked_g = masks_for_g * g
            # masked_gt = masks_for_g * gt

            l1 = l1loss(half_g * half_bypass_mouth, half_gt * half_bypass_mouth).reshape(
                (B, hparams.syncnet_T * 3 * hparams.half_img_size * hparams.img_size)).mean(1)

            ssim = ms_ssim_loss(gt * bypass_mouth, g * bypass_mouth, B, hparams.syncnet_T)
            rec_loss = hparams.l1_wt * \
                l1 + hparams.ssim_wt * ssim

            loss = (hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss +
                    landmarks_loss * hparams.landmarks_wt + (1. - hparams.syncnet_wt - hparams.disc_wt) * rec_loss +
                    blurs_loss * hparams.blurs_wt)
            loss = (loss * weights / K).mean()
            loss.backward()

            if global_step % K == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, norm_type=2.0)
                optimizer.step()
                optimizer.zero_grad()

            pred = disc(half_gt, half_x)
            disc_batch_size = pred.size(0)
            y_real = torch.ones((disc_batch_size, 1, 10, 22),
                                dtype=torch.float32, device=device)
            disc_real_loss = F.binary_cross_entropy(
                pred, y_real, reduction='none').mean(2).mean(2)

            pred = disc(half_g.detach(), half_x)
            y_fake = torch.zeros((disc_batch_size, 1, 10, 22),
                                 dtype=torch.float32, device=device)
            disc_fake_loss = F.binary_cross_entropy(
                pred, y_fake, reduction='none').mean(2).mean(2)

            disc_loss = (disc_real_loss + disc_fake_loss) / K / 2.
            disc_loss = (disc_loss.reshape((B, hparams.syncnet_T)).mean(1) * weights).mean()
            disc_loss.backward()

            if global_step % K == 0:
                torch.nn.utils.clip_grad_norm_(
                    disc.parameters(), 1.0, norm_type=2.0)
                disc_optimizer.step()
                disc_optimizer.zero_grad()

                # torch.nn.utils.clip_grad_norm_(
                #     syncnet.parameters(), 1.0, norm_type=2.0)
                # syncnet_optimizer.step()
                # syncnet_optimizer.zero_grad()

            if global_step % checkpoint_interval == 0:
                save_sample_images(x, g, gt, global_step,
                                   checkpoint_dir,
                                   landmarks_g,
                                   landmarks_gt,
                                   summary_writer)

            # Logs
            global_step += 1
            # cur_session_steps = global_step - resumed_step

            running_disc_real_loss += disc_real_loss.detach().mean()
            running_disc_fake_loss += disc_fake_loss.detach().mean()
            running_disc_loss += disc_loss.detach()
            running_target_loss += loss.detach()
            running_l1_loss += l1.detach().mean()
            running_ssim_loss += ssim.detach().mean()
            running_sync_loss += sync_loss.detach().mean()
            running_perceptual_loss += perceptual_loss.detach().mean()

            running_sync_real_loss += sync_real_loss.detach().mean()
            running_landmarks_loss += landmarks_loss.detach().mean()
            running_blurs_loss += blurs_loss.detach().mean()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)
                save_checkpoint(disc, disc_optimizer, global_step,
                                checkpoint_dir, global_epoch, prefix='disc_')
                # save_checkpoint(syncnet, syncnet_optimizer, global_step,
                #                 checkpoint_dir, global_epoch, prefix='sync_')

            if global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                    eval_model(
                        test_data_loader, global_step, device, model, disc, syncnet, summary_writer=summary_writer)

            next_step = step + 1

            if global_step % K == 0:

                _l1 = running_l1_loss.item() / next_step
                _ssim = running_ssim_loss.item() / next_step
                _sync = running_sync_loss.item() / next_step
                _perc = running_perceptual_loss.item() / next_step
                _disc_fake = running_disc_fake_loss.item() / next_step
                _disc_real = running_disc_real_loss.item() / next_step
                _disc = running_disc_loss.item() / next_step * K
                _target = running_target_loss.item() / next_step * K

                _sync_real = running_sync_real_loss.item() / next_step

                _landmarks = running_landmarks_loss.item() / next_step
                _blurs = running_blurs_loss.item() / next_step

                if _disc < 0.7 and _perc < 1.0:
                    if hparams.disc_wt != original_disc_wt:
                        print(
                            "discriminator is trustable now, set it back to:", original_disc_wt)
                        hparams.set_hparam('disc_wt', original_disc_wt)
                elif running_l1_loss.item() / next_step > 0.05 and hparams.disc_wt != 0.001 and _disc > 0.8:
                    print("discriminator is not trustable, set weight to 0.001")
                    hparams.set_hparam('disc_wt', 0.001)

                if _sync_real < 0.4 and _sync < 0.35:
                    if hparams.syncnet_wt != origin_syncnet_wt:
                        print(
                            "syncnet is trustable now, set it back to:", origin_syncnet_wt)
                        hparams.set_hparam('syncnet_wt', origin_syncnet_wt)
                elif running_l1_loss.item() / next_step > 0.05 and hparams.syncnet_wt != 0.001:
                    print("syncnet is not trustable, set weight to 0.001")
                    hparams.set_hparam('syncnet_wt', 0.001)

                prog_bar.set_description(
                    'L1: {:.3f}, SSIM: {:.3f}, Land: {:.4f}, Sync: {:.3f}, Percep: {:.3f}, Blurs: {:.3f} | Fake: {:.3f}, Real: {:.3f}, Disc: {:.3f} | SyncReal: {:.3f} | Target: {:.3f}'.format(
                        _l1,
                        _ssim,
                        _landmarks,
                        _sync,
                        _perc,
                        _blurs,
                        _disc_fake,
                        _disc_real,
                        _disc,
                        _sync_real,
                        _target,
                    ))
                if summary_writer is not None:
                    summary_writer.add_scalar("Train/Gen/L1", _l1, global_step)
                    summary_writer.add_scalar(
                        "Train/Gen/MS-SSIM", _ssim, global_step)
                    summary_writer.add_scalar("Train/Gen/Sync", _sync, global_step)
                    summary_writer.add_scalar(
                        "Train/Gen/Percep", _perc, global_step)
                    summary_writer.add_scalar(
                        "Train/Gen/Target", _target, global_step)
                    summary_writer.add_scalar(
                        "Train/Gen/Land", _landmarks, global_step)
                    summary_writer.add_scalar(
                        "Train/Gen/Blurs", _blurs, global_step)
                    summary_writer.add_scalar(
                        "Train/Disc/Fake", _disc_fake, global_step)
                    summary_writer.add_scalar(
                        "Train/Disc/Real", _disc_real, global_step)
                    summary_writer.add_scalar(
                        "Train/Disc/Target", _disc, global_step)
                    summary_writer.add_scalar(
                        "Train/Sync/Real", _sync_real, global_step)

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, disc, syncnet, summary_writer=None):
    eval_steps = 300
    print('Evaluating for {} steps'.format(eval_steps))
    running_sync_loss, running_disc_real_loss, \
        running_disc_fake_loss, running_disc_train_loss, \
        running_perceptual_loss, running_target_loss, \
        running_l1_loss, running_ssim_loss, \
        running_sync_real_loss, \
        running_landmarks_loss, running_blurs_loss = [
        ], [], [], [], [], [], [], [], [], [], []

    for step, (x, indiv_mels, mel, gt, landmarks_gt, blurs_gt, weights, masks) in enumerate((test_data_loader)):
        B = x.size(0)
        if B == 1:
            continue
        model.eval()
        disc.eval()
        syncnet.eval()

        x = x.to(device)
        half_x = x[:, :, :, hparams.half_img_size:]
        mel = mel.to(device)
        indiv_mels = indiv_mels.to(device)
        gt = gt.to(device)
        landmarks_gt = landmarks_gt.to(device)
        weights = weights.to(device)
        blurs_gt = blurs_gt.to(device)
        masks = masks.to(device)
        bypass_mouth_masks = 1.0 - masks
        half_bypass_mouth_masks = bypass_mouth_masks[:, :, :, hparams.half_img_size:]

        half_g, landmarks_g = model(indiv_mels, x)
        upper_gt = gt[:, :, :, :hparams.half_img_size]
        g = torch.cat([upper_gt, half_g], dim=3)

        landmarks_loss = get_landmarks_loss(landmarks_g, landmarks_gt)
        half_gt = gt[:, :, :, hparams.half_img_size:]

        pred = disc(half_gt, half_x)
        disc_batch_size = pred.size(0)
        y_real = torch.ones((disc_batch_size, 1, 10, 22),
                            dtype=torch.float32, device=device)
        disc_real_loss = F.binary_cross_entropy(
            pred, y_real)

        pred = disc(half_g, half_x)
        y_fake = torch.zeros((disc_batch_size, 1, 10, 22),
                             dtype=torch.float32, device=device)
        disc_fake_loss = F.binary_cross_entropy(
            pred, y_fake)

        sync_real_loss = get_sync_loss(syncnet, mel, half_gt, expect_true=True)
        sync_loss = get_sync_loss(syncnet, mel, half_g)

        sync_loss = torch.maximum(sync_loss * weights - sync_real_loss, torch.zeros((B,), dtype=torch.float32, device=device))

        blurs_loss = get_blurs_loss(half_g, blurs_gt, B)

        perceptual_loss = disc.perceptual_forward(half_g, half_x)
        perceptual_loss = perceptual_loss.reshape((B, hparams.syncnet_T)).mean(1)

        l1 = l1loss(half_g * half_bypass_mouth_masks, half_gt * half_bypass_mouth_masks).reshape(
                (B, hparams.syncnet_T * 3 * hparams.half_img_size * hparams.img_size)).mean(1)

        ssim = ms_ssim_loss(gt * bypass_mouth_masks, g * bypass_mouth_masks, B, hparams.syncnet_T)
        rec_loss = hparams.l1_wt * \
            l1 + hparams.ssim_wt * ssim

        loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
            landmarks_loss * hparams.landmarks_wt + \
            blurs_loss * hparams.blurs_wt + \
            (1. - hparams.syncnet_wt - hparams.disc_wt) * rec_loss
        loss = (loss * weights).mean()

        real = disc_real_loss.detach().mean()
        fake = disc_fake_loss.detach().mean()
        running_disc_real_loss.append(real)
        running_disc_fake_loss.append(fake)
        running_disc_train_loss.append((real + fake) / 2.0)

        running_target_loss.append(loss.detach())
        running_l1_loss.append(l1.detach().mean())
        running_ssim_loss.append(ssim.detach().mean())
        running_sync_loss.append(sync_loss.detach().mean())
        running_perceptual_loss.append(perceptual_loss.detach().mean())

        real = sync_real_loss.detach().mean()
        running_sync_real_loss.append(real)
        running_landmarks_loss.append(landmarks_loss.detach().mean())

        running_blurs_loss.append(blurs_loss.detach().mean())

        if step > eval_steps:
            break
    _l1 = np.mean([loss.item() for loss in running_l1_loss])
    _ssim = np.mean([loss.item() for loss in running_ssim_loss])
    _sync = np.mean([loss.item() for loss in running_sync_loss])
    _perc = np.mean([loss.item() for loss in running_perceptual_loss])
    _blurs = np.mean([loss.item() for loss in running_blurs_loss])
    _disc_fake = np.mean([loss.item() for loss in running_disc_fake_loss])
    _disc_real = np.mean([loss.item() for loss in running_disc_real_loss])
    _disc = np.mean([loss.item() for loss in running_disc_train_loss])
    _target = np.mean([loss.item() for loss in running_target_loss])
    _sync_real = np.mean([loss.item() for loss in running_sync_real_loss])
    _landmarks = np.mean([loss.item() for loss in running_landmarks_loss])
    print('L1: {:.3f}, SSIM: {:.3f}, Land: {:.4f}, Sync: {:.3f}, Percep: {:.3f}, Blurs: {:.3f} | Fake: {:.3f}, Real: {:.3f}, Disc: {:.3f} | SyncReal: {:.3f} | Target: {:.3f}'.format(
        _l1,
        _ssim,
        _landmarks,
        _sync,
        _perc,
        _blurs,
        _disc_fake,
        _disc_real,
        _disc,
        _sync_real,
        _target))
    if summary_writer is not None:
        summary_writer.add_scalar("Evaluation/Gen/L1", _l1, global_step)
        summary_writer.add_scalar("Evaluation/Gen/MS-SSIM", _ssim, global_step)
        summary_writer.add_scalar("Evaluation/Gen/Sync", _sync, global_step)
        summary_writer.add_scalar("Evaluation/Gen/Land", _landmarks, global_step)
        summary_writer.add_scalar("Evaluation/Gen/Percep", _perc, global_step)
        summary_writer.add_scalar("Evaluation/Gen/Blurs", _blurs, global_step)
        summary_writer.add_scalar("Evaluation/Gen/Target", _target, global_step)
        summary_writer.add_scalar("Evaluation/Disc/Fake", _disc_fake, global_step)
        summary_writer.add_scalar("Evaluation/Disc/Real", _disc_real, global_step)
        summary_writer.add_scalar(
            "Evaluation/Disc/Target", _disc, global_step)
        summary_writer.add_scalar(
            "Evaluation/Sync/Real", _sync_real, global_step)

    return _sync


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
    if not reset_optimizer and optimizer is not None:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"] + 1

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
                            help='Load the pre-trained Expert discriminator', type=str)

        parser.add_argument(
            '--checkpoint_path', help='Resume generator from this checkpoint', default=None, type=str)
        parser.add_argument('--disc_checkpoint_path',
                            help='Resume quality disc from this checkpoint', default=None, type=str)
        parser.add_argument('--train_limit', type=int,
                            required=False, default=0)
        parser.add_argument('--val_limit', type=int, required=False, default=0)
        parser.add_argument('--filelists_dir',
                            help='Specify filelists directory', type=str, default='filelists')
        parser.add_argument('--K',
                            help='Delay update', type=int, default=1)
        parser.add_argument('--reset_optimizer',
                            help='Reset optimizer or not', action='store_true')
        parser.add_argument('--reset_disc_optimizer',
                            help='Reset disc optimizer or not', action='store_true')
        parser.add_argument('--hparams',
                            help='specify hparams file, default is None, this overwrite is after the env overwrite',
                            type=str, default=None)
        parser.add_argument('--inception',
                            help='Use InceptionV3 Network as discriminator', action='store_true')
        parser.add_argument('--shufflenet',
                            help='Use ShuffleNetV2 Network as syncnet', action='store_true')
        parser.add_argument('--use_syncnet_weights',
                            help='Use Syncnet Weights for training', action='store_true')
        parser.add_argument('--logdir',
                            help='Tensorboard logdir', default=None, type=str)
        args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir

    now = datetime.now()
    logdir = os.path.join(
        checkpoint_dir, "log-wav2lip-{}".format(now.strftime("%Y-%m-%d-%H_%M")))
    if args.logdir is not None:
        logdir = args.logdir
    print("log at: {}".format(logdir))
    summary_writer = SummaryWriter(logdir)
    summary_writer.add_text("parameters", hparams.to_json())

    if args.hparams is None:
        hparams_dump_path = os.path.join(
            checkpoint_dir, "hparams-wav2lip-{}.json".format(now.strftime("%Y-%m-%d-%H_%M")))
        hparams.to_json(hparams_dump_path)
    else:
        hparams.overwrite_by_json(args.hparams)

    # Dataset and Dataloader setup
    train_dataset = Wav2LipDataset(
        'train', args.data_root,
        sampling_half_window_size_seconds=hparams.sampling_half_window_size_seconds,
        limit=args.train_limit,
        filelists_dir=args.filelists_dir,
        img_augment=hparams.img_augment,
        inner_shuffle=False,
        use_syncnet_weights=args.use_syncnet_weights)
    test_dataset = Wav2LipDataset(
        'val', args.data_root,
        sampling_half_window_size_seconds=hparams.sampling_half_window_size_seconds,
        img_augment=False,
        limit=300,  # val steps
        filelists_dir=args.filelists_dir,
        inner_shuffle=False,
        use_syncnet_weights=False)

    def worker_init_fn(i):
        seed = int(time()) + i * 100
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        return

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        worker_init_fn=worker_init_fn,
        shuffle=True)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=2,
        worker_init_fn=worker_init_fn)

    # Model
    model = Wav2Lip().to(device)
    if args.inception:
        print("**** Enable Inception V3 as discriminator ****")
        disc = InceptionV3_disc(
            pretrained=False).to(device)
    else:
        disc = Wav2Lip_disc_qual().to(device)

    if args.shufflenet:
        print("**** Enable ShuffleNet V2 1.0 as syncnet ****")
        syncnet = SyncNet_shuffle_color().to(device)
    else:
        syncnet = SyncNet().to(device)

    print('total trainable params {}'.format(sum(p.numel()
          for p in model.parameters() if p.requires_grad)))
    print('total DISC trainable params {}'.format(sum(p.numel()
          for p in disc.parameters() if p.requires_grad)))
    print('total SYNC trainable params {}'.format(sum(p.numel()
          for p in syncnet.parameters() if p.requires_grad)))
    if args.reset_optimizer:
        print("reset optimizer")
    if args.reset_disc_optimizer:
        print("reset disc optimizer")

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=hparams.initial_learning_rate, betas=(0.5, 0.999),
        amsgrad=hparams.opt_amsgrad, weight_decay=hparams.opt_weight_decay)
    disc_optimizer = optim.Adam(
        [p for p in disc.parameters() if p.requires_grad],
        lr=hparams.disc_initial_learning_rate, betas=(0.5, 0.999),
        amsgrad=hparams.disc_opt_amsgrad, weight_decay=hparams.disc_opt_weight_decay)

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model,
                        optimizer, reset_optimizer=args.reset_optimizer,
                        overwrite_global_states=not args.reset_optimizer)

    if args.disc_checkpoint_path is not None:
        load_checkpoint(args.disc_checkpoint_path, disc, disc_optimizer,
                        reset_optimizer=args.reset_disc_optimizer,
                        overwrite_global_states=not args.reset_disc_optimizer)

    if args.syncnet_checkpoint_path is not None:
        load_checkpoint(args.syncnet_checkpoint_path, syncnet, None,
                        overwrite_global_states=False)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    model = nn.DataParallel(model)
    # Train!
    avg_fully_ssim_loss = train(
        device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
        syncnet,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=hparams.checkpoint_interval,
        nepochs=hparams.nepochs, K=args.K,
        summary_writer=summary_writer)
    return avg_fully_ssim_loss


if __name__ == "__main__":
    main()
