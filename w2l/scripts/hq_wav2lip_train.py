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
from torch.utils.tensorboard import SummaryWriter
import random
from time import time
from datetime import datetime

global_step = 0
global_epoch = 0

syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False
syncnet.eval()


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


def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    # g_landmarks: (B, T, landmarks_points_len, 2)
    x = (x.detach().cpu().numpy().transpose(
        0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(
        0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(
        0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    # gt: (B, T, H, W, 3)
    refs, inps = x[..., 3:], x[..., :3]

    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder):
        os.mkdir(folder)

    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


l1loss = nn.L1Loss()


def get_landmarks_loss(g_landmarks, gt_landmarks):
    axis_delta = g_landmarks[:, :, :, 0] - gt_landmarks[:, :,
                                                        :, 0] + g_landmarks[:, :, :, 1] - gt_landmarks[:, :, :, 1]
    square_mean = torch.mean(torch.mean(axis_delta ** 2, dim=0), dim=-1)
    loss_sum_T = torch.sum(square_mean)
    return loss_sum_T / hparams.syncnet_T


# resize_for_sync = torchvision.transforms.Resize((48, 96))


def get_sync_loss(mel, half_g):
    # g: B x 3 x T x H x W, g should be masked
    half_g = torch.cat([half_g[:, :, i]
                       for i in range(hparams.syncnet_T)], dim=1)
    # half_g = resize_for_sync(half_g)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, half_g)
    y = torch.ones((half_g.size(0), 1), dtype=torch.float32, device=device)
    return cosine_loss(a, v, y)


def train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None, K=1,
          summary_writer=None):
    global global_step, global_epoch
    # resumed_step = global_step

    original_disc_wt = hparams.disc_wt
    half_img_size = hparams.img_size // 2
    expand_y_start = half_img_size - max(10, 168 - half_img_size)

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
        print("epoch: {}, lr: {}, disc_lr: {}".format(global_epoch, lr, disc_lr))
        if summary_writer is not None:
            summary_writer.add_scalar("Train/Wav2Lip-LR", lr, global_step)
            summary_writer.add_scalar("Train/Disc-LR", disc_lr, global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in disc_optimizer.param_groups:
            param_group['lr'] = disc_lr

        running_sync_loss, running_rec_loss, running_perceptual_loss = 0., 0., 0.
        running_disc_real_loss, running_disc_fake_loss, running_target_loss = 0., 0., 0.
        running_l1_loss, running_ssim_loss, running_disc_loss = 0., 0., 0.

        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, indiv_mels, mel, gt, masks) in prog_bar:
            B = x.size(0)
            if B == 1:
                continue
            disc.train()
            model.train()

            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)
            # masks = masks.to(device)
            half_gt = gt[:, :, :, hparams.img_size // 2:]
            # Train generator now. Remove ALL grads.

            half_g = model(indiv_mels, x)

            sync_loss = get_sync_loss(mel, half_g)

            perceptual_loss = disc.perceptual_forward(half_g)

            # masks_for_g = masks.permute((0, 2, 1, 3, 4))
            ssim_gt = gt[:, :, :, expand_y_start:]
            ssim_g = torch.cat(
                (gt[:, :, :, expand_y_start:half_img_size], half_g), dim=3)

            # debug_dump(ssim_gt, 'temp/ssim_gt.png')
            # debug_dump(ssim_g, 'temp/ssim_g.png')
            # refs, inps = x[:, 3:], x[:, :3]
            # debug_dump(refs, 'temp/refs.png')
            # debug_dump(inps, 'temp/inps.png')

            # masked_g = masks_for_g * g
            # masked_gt = masks_for_g * gt

            l1 = l1loss(half_g, half_gt)

            ssim = ms_ssim_loss(ssim_g, ssim_gt)
            rec_loss = hparams.l1_wt * \
                l1 + hparams.ssim_wt * ssim

            loss = (hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss +
                    (1. - hparams.syncnet_wt - hparams.disc_wt) * rec_loss)
            loss /= K
            loss.backward()

            if global_step % K == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, norm_type=2.0)
                optimizer.step()
                optimizer.zero_grad()

            pred = disc(half_gt)
            disc_batch_size = pred.size(0)
            y_real = torch.ones((disc_batch_size, 1),
                                dtype=torch.float32, device=device)
            disc_real_loss = F.binary_cross_entropy(
                pred, y_real)

            pred = disc(half_g.detach())
            y_fake = torch.zeros((disc_batch_size, 1),
                                 dtype=torch.float32, device=device)
            disc_fake_loss = F.binary_cross_entropy(
                pred, y_fake)

            disc_loss = (disc_real_loss + disc_fake_loss) / K / 2.
            disc_loss.backward()

            if global_step % K == 0:
                torch.nn.utils.clip_grad_norm_(
                    disc.parameters(), 1.0, norm_type=2.0)
                disc_optimizer.step()
                disc_optimizer.zero_grad()

            if global_step % checkpoint_interval == 0:
                g_zeros = torch.zeros_like(half_g)
                g = torch.cat((g_zeros, half_g), dim=3)
                save_sample_images(x, g, gt, global_step,
                                   checkpoint_dir)

            # Logs
            global_step += 1
            # cur_session_steps = global_step - resumed_step

            running_disc_real_loss += disc_real_loss.detach()
            running_disc_fake_loss += disc_fake_loss.detach()
            running_disc_loss += disc_loss.detach()
            running_target_loss += loss.detach()
            running_rec_loss += rec_loss.detach()
            running_l1_loss += l1.detach()
            running_ssim_loss += ssim.detach()
            running_sync_loss += sync_loss.detach()
            running_perceptual_loss += perceptual_loss.detach()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)
                save_checkpoint(disc, disc_optimizer, global_step,
                                checkpoint_dir, global_epoch, prefix='disc_')

            if global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                    average_sync_loss = eval_model(
                        test_data_loader, global_step, device, model, disc, summary_writer=summary_writer)

                    if average_sync_loss < .6:
                        print("set syncnet_wt as", 0.03)
                        hparams.set_hparam('syncnet_wt', 0.03)

            next_step = step + 1

            if global_step % K == 0:
                if running_disc_loss.item() / next_step * K < 2.0:
                    if hparams.disc_wt != original_disc_wt:
                        print(
                            "discriminator is trustable now, set it back to:", original_disc_wt)
                        hparams.set_hparam('disc_wt', original_disc_wt)
                elif running_l1_loss.item() / next_step > 0.03:
                    print("discriminator is not trustable, set weight to 0.")
                    hparams.set_hparam('disc_wt', 0.)

                _l1 = running_l1_loss.item() / next_step
                _ssim = running_ssim_loss.item() / next_step
                _rec = running_rec_loss.item() / next_step
                _sync = running_sync_loss.item() / next_step
                _perc = running_perceptual_loss.item() / next_step
                _disc_fake = running_disc_fake_loss.item() / next_step
                _disc_real = running_disc_real_loss.item() / next_step
                _target = running_target_loss.item() / next_step * K
                prog_bar.set_description(
                    'L1: {:03f}, SSIM: {:03f}, Rec: {:03f}, Sync: {:03f}, Percep: {:03f} | Fake: {:03f}, Real: {:03f} | Target: {:03f}'.format(
                        _l1,
                        _ssim,
                        _rec,
                        _sync,
                        _perc,
                        _disc_fake,
                        _disc_real,
                        _target,
                    ))
                if summary_writer is not None:
                    summary_writer.add_scalar("Train/L1", _l1, global_step)
                    summary_writer.add_scalar(
                        "Train/MS-SSIM", _ssim, global_step)
                    summary_writer.add_scalar("Train/Rec", _rec, global_step)
                    summary_writer.add_scalar("Train/Sync", _sync, global_step)
                    summary_writer.add_scalar(
                        "Train/Percep", _perc, global_step)
                    summary_writer.add_scalar(
                        "Train/Fake", _disc_fake, global_step)
                    summary_writer.add_scalar(
                        "Train/Real", _disc_real, global_step)
                    summary_writer.add_scalar(
                        "Train/Target", _target, global_step)

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, disc, summary_writer=None):
    eval_steps = 300
    print('Evaluating for {} steps'.format(eval_steps))
    running_sync_loss, recon_losses, running_disc_real_loss, \
        running_disc_fake_loss, running_perceptual_loss, running_target_loss, \
        running_l1_loss, running_ssim_loss = [], [], [], [], [], [], [], []

    half_img_size = hparams.img_size // 2
    expand_y_start = half_img_size - max(10, 168 - half_img_size)
    for step, (x, indiv_mels, mel, gt, masks) in enumerate((test_data_loader)):
        B = x.size(0)
        if B == 1:
            continue
        model.eval()
        disc.eval()

        x = x.to(device)
        mel = mel.to(device)
        indiv_mels = indiv_mels.to(device)
        gt = gt.to(device)
        # masks = masks.to(device)
        half_gt = gt[:, :, :, half_img_size:]

        pred = disc(half_gt)
        disc_batch_size = pred.size(0)
        y_real = torch.ones((disc_batch_size, 1),
                            dtype=torch.float32, device=device)
        disc_real_loss = F.binary_cross_entropy(
            pred, y_real)

        half_g = model(indiv_mels, x)
        # g_zeros = torch.zeros_like(half_g)
        # g = torch.cat((g_zeros, half_g), dim=3)

        pred = disc(half_g)
        y_fake = torch.zeros((disc_batch_size, 1),
                             dtype=torch.float32, device=device)
        disc_fake_loss = F.binary_cross_entropy(
            pred, y_fake)

        sync_loss = get_sync_loss(mel, half_g)

        perceptual_loss = disc.perceptual_forward(half_g)

        # masks_for_g = masks.permute((0, 2, 1, 3, 4))
        # masked_g = masks_for_g * g
        # masked_gt = masks_for_g * gt

        ssim_gt = gt[:, :, :, expand_y_start:]
        ssim_g = torch.cat(
            (gt[:, :, :, expand_y_start:half_img_size], half_g), dim=3)

        l1 = l1loss(half_g, half_gt)
        ssim = ms_ssim_loss(ssim_gt, ssim_g)
        rec_loss = hparams.l1_wt * \
            l1 + hparams.ssim_wt * ssim

        loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
            (1. - hparams.syncnet_wt - hparams.disc_wt) * rec_loss

        running_disc_real_loss.append(disc_real_loss.detach())
        running_disc_fake_loss.append(disc_fake_loss.detach())
        running_target_loss.append(loss.detach())
        recon_losses.append(rec_loss.detach())
        running_l1_loss.append(l1.detach())
        running_ssim_loss.append(ssim.detach())
        running_sync_loss.append(sync_loss.detach())
        running_perceptual_loss.append(perceptual_loss.detach())

        if step > eval_steps:
            break
    _l1 = np.mean([loss.item() for loss in running_l1_loss])
    _ssim = np.mean([loss.item() for loss in running_ssim_loss])
    _rec = np.mean([loss.item() for loss in recon_losses])
    _sync = np.mean([loss.item() for loss in running_sync_loss])
    _perc = np.mean([loss.item() for loss in running_perceptual_loss])
    _disc_fake = np.mean([loss.item() for loss in running_disc_fake_loss])
    _disc_real = np.mean([loss.item() for loss in running_disc_real_loss])
    _target = np.mean([loss.item() for loss in running_target_loss])
    print('L1: {:03f}, SSIM: {:03f}, Rec: {:04f}, Sync: {:04f}, Percep: {:04f} | Fake: {:04f}, Real: {:04f} | Target: {:04f}'.format(
        _l1,
        _ssim,
        _rec,
        _sync,
        _perc,
        _disc_fake,
        _disc_real,
        _target))
    if summary_writer is not None:
        summary_writer.add_scalar("Evaluation/L1", _l1, global_step)
        summary_writer.add_scalar("Evaluation/MS-SSIM", _ssim, global_step)
        summary_writer.add_scalar("Evaluation/Rec", _rec, global_step)
        summary_writer.add_scalar("Evaluation/Sync", _sync, global_step)
        summary_writer.add_scalar("Evaluation/Percep", _perc, global_step)
        summary_writer.add_scalar("Evaluation/Fake", _disc_fake, global_step)
        summary_writer.add_scalar("Evaluation/Real", _disc_real, global_step)
        summary_writer.add_scalar("Evaluation/Target", _target, global_step)

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
    if not reset_optimizer:
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
        parser.add_argument('--K',
                            help='Delay update', type=int, default=1)
        parser.add_argument('--reset_optimizer',
                            help='Reset optimizer or not', action='store_true')
        parser.add_argument('--reset_disc_optimizer',
                            help='Reset disc optimizer or not', action='store_true')
        parser.add_argument('--hparams',
                            help='specify hparams file, default is None, this overwrite is after the env overwrite',
                            type=str, default=None)
        args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir

    now = datetime.now()
    log_dir = os.path.join(
        checkpoint_dir, "log-wav2lip-{}".format(now.strftime("%Y-%m-%d-%H_%M")))
    print("log at: {}".format(log_dir))
    summary_writer = SummaryWriter(log_dir)

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
        img_augment=hparams.img_augment)
    test_dataset = Wav2LipDataset(
        'val', args.data_root,
        sampling_half_window_size_seconds=hparams.sampling_half_window_size_seconds,
        img_augment=False,
        limit=300,  # val steps
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
        train_dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        worker_init_fn=worker_init_fn)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=2,
        worker_init_fn=worker_init_fn)

    # Model
    model = Wav2Lip().to(device)
    disc = Wav2Lip_disc_qual().to(device)

    print('total trainable params {}'.format(sum(p.numel()
          for p in model.parameters() if p.requires_grad)))
    print('total DISC trainable params {}'.format(sum(p.numel()
          for p in disc.parameters() if p.requires_grad)))
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

    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True,
                    overwrite_global_states=False)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    model = nn.DataParallel(model)
    # Train!
    avg_fully_ssim_loss = train(
        device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=hparams.checkpoint_interval,
        nepochs=hparams.nepochs, K=args.K,
        summary_writer=summary_writer)
    return avg_fully_ssim_loss


if __name__ == "__main__":
    main()
