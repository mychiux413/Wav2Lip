import argparse
import os
import numpy as np
from torch.utils import data as data_utils
from w2l.models.syncnet import SyncNet_color as SyncNet
from w2l.utils.data import Dataset
import random
import torch
from torch import nn
from w2l.hparams import hparams as hp
from w2l.utils import audio
from tqdm import tqdm
from glob import glob
import cv2
from functools import partial
from PIL import Image


class SyncnetDataset(Dataset):
    def __init__(self, data_root, only_true_image=True, img_size=96):
        self.all_videos = []
        for dirname in os.listdir(data_root):
            dirpath = os.path.join(data_root, dirname)
            if not os.path.isdir(dirpath):
                continue
            for vid_dirname in os.listdir(dirpath):
                video_path = os.path.join(dirpath, vid_dirname)
                wavpath = os.path.join(video_path, "audio.wav")
                if len(os.listdir(video_path)) < 3 * hp.syncnet_T + 2:
                    print("insufficient files of dir:", vid_dirname)
                    continue
                if not os.path.exists(wavpath):
                    print("skip missing audio of:", vid_dirname)
                    continue
                self.all_videos.append(video_path)

        self.img_names = {
            vidname: sorted(
                glob(os.path.join(vidname, '*.png')),
                key=lambda name: int(os.path.basename(name).split('.')[0])) for vidname in self.all_videos
        }

        self.orig_mels = {}
        for vidname in tqdm(self.all_videos, desc="load mels"):
            mel_path = os.path.join(vidname, "mel.npy")
            wavpath = os.path.join(vidname, "audio.wav")
            if os.path.exists(mel_path):
                try:
                    orig_mel = np.load(mel_path)
                except Exception as err:
                    print(err)
                    wav = audio.load_wav(wavpath, hp.sample_rate)
                    orig_mel = audio.melspectrogram(wav).T
                    np.save(mel_path, orig_mel)
            else:
                wav = audio.load_wav(wavpath, hp.sample_rate)
                orig_mel = audio.melspectrogram(wav).T
                np.save(mel_path, orig_mel)
            self.orig_mels[vidname] = orig_mel
        self.data_root = data_root
        self.inner_shuffle = False
        self.sampling_half_window_size_seconds = 1e10

        # 實驗發現, 只要是wrong image, model的分辨能力都很好, 因此不需要sampling wrong image
        self.only_true_image = only_true_image
        self.img_size = img_size

    def __getitem__(self, idx):
        while 1:
            vidname = self.get_vidname(idx)

            img_names = self.img_names[vidname]
            if len(img_names) <= 3 * self.syncnet_T:
                idx += 1
                idx %= len(self)
                continue

            img_name, wrong_img_name = self.sample_right_wrong_images(
                img_names)

            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            if self.only_true_image or random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                idx += 1
                idx %= len(self)
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(
                        img, (self.img_size, self.img_size))
                except Exception as _:  # noqa: F841
                    all_read = False
                    break

                window.append(img)

            if not all_read:
                idx += 1
                idx %= len(self)
                continue

            orig_mel = self.orig_mels[vidname]
            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != self.syncnet_mel_step_size):
                idx += 1
                idx %= len(self)
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y, os.path.relpath(vidname, self.data_root)


logloss = nn.BCELoss(reduction='none')


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


def evaluate_datasets_losses(syncnet_checkpoint_path, img_size, data_root, epochs=5):

    sync_losses_path = os.path.join(data_root, "synclosses.npy")

    # **** load cache file of synclosses or bypass ****
    if os.path.exists(sync_losses_path):
        try:
            print("load cache file of synclosses losses:", sync_losses_path)
            losses = np.load(sync_losses_path, allow_pickle=True).tolist()

            epochs_in_losses = max([len(v) for v in losses.values()])
            if epochs_in_losses >= epochs:
                stat_losses = {}
                for vidname, loss in losses.items():
                    stat_losses[vidname] = (np.mean(losses[vidname]),
                                            np.std(losses[vidname]))
                return stat_losses
        except Exception as err:
            print(err)
    # *****************************************

    hp.set_hparam('img_size', img_size)
    device = 'cuda:0'
    sync_model = SyncNet().to(device)
    checkpoint = torch.load(syncnet_checkpoint_path)
    sync_model.load_state_dict(checkpoint["state_dict"])
    test_dataset = SyncnetDataset(
        data_root, only_true_image=True, img_size=hp.img_size)
    data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hp.syncnet_batch_size,
        num_workers=hp.num_workers,
        persistent_workers=True)

    sync_model.eval()
    losses = {}
    with torch.no_grad():
        for epoch in range(epochs):
            for x, mel, y, vidnames in tqdm(data_loader, desc="[epoch {}] evaluate sync loss".format(epoch)):
                for vidname in vidnames:
                    if vidname not in losses:
                        losses[vidname] = []
                x = x.to(device)
                mel = mel.to(device)
                a, v = sync_model(mel, x)
                y = y.to(device)
                loss = cosine_loss(a, v, y)[:, 0].to('cpu').numpy()
                for vidname, l in zip(vidnames, loss):
                    losses[vidname].append(l)
    np.save(sync_losses_path, losses, allow_pickle=True)

    stat_losses = {}
    for vidname, loss in losses.items():
        stat_losses[vidname] = (np.mean(losses[vidname]),
                                np.std(losses[vidname]))
    return stat_losses


def get_min_img_size(path):
    with Image.open(path) as im:
        width, height = im.size
    return min(width, height)


def to_sorted_stats_landmarks(tup, delta=True):
    path, dic = tup
    values = []
    for k, v in sorted(dic.items(), key=lambda item: int(item[0].split('.')[0])):
        values.append(v)
    rows = np.array(values)
    if delta:
        rows = np.diff(rows, axis=0)
    mean = np.mean(rows, axis=0)
    std = np.std(rows, axis=0)
    return path, np.stack([mean, std], axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--train_ratio', type=float,
                        required=False, default=0.95)
    parser.add_argument('--train_limit', type=int, required=False, default=0)
    parser.add_argument('--val_limit', type=int, required=False, default=0)
    parser.add_argument("--syncnet_checkpoint_path",
                        help='Load the pre-trained Expert discriminator', required=False, default=None)
    parser.add_argument('--cosine_loss_mean_max_q',
                        help='Pass the loss of datasets under specified mean quantile', default=0.9, type=float)
    parser.add_argument('--cosine_loss_std_max_q',
                        help='Pass the loss of datasets under specified std quantile', default=0.9, type=float)
    parser.add_argument('--lip_mean_cut_q',
                        help='', default=0.9, type=float)
    parser.add_argument('--lip_std_cut_q',
                        help='', default=0.9, type=float)
    parser.add_argument('--filter_outbound_lip',
                        help='', action='store_true')
    parser.add_argument('--cosine_loss_epoch',
                        help='Specify the epoch to evaluate cosine loss', default=10, type=int)
    parser.add_argument('--syncnet_img_size',
                        help='Image Size of Syncnet', default=96, type=int)
    parser.add_argument('--min_img_size',
                        help='', default=0, type=int)
    parser.add_argument('--min_mean_blur_score',
                        help='', default=0, type=float)
    parser.add_argument('--blur_score_q_cut',
                        help='', default=1.0, type=float)
    parser.add_argument('--filelists_dir',
                        help='Specify filelists directory', type=str, default='filelists')
    parser.add_argument('--include_train_dirs',
                        help='for dirs into training datasets with comma seperated', default="", type=str)

    args = parser.parse_args()

    assert os.path.exists(args.data_root)
    assert args.train_ratio < 1.0
    assert args.train_ratio > 0.0
    include_train_dirs = args.include_train_dirs.split(",")

    valid_vidnames = set()
    if args.syncnet_checkpoint_path is not None:
        stat_losses = evaluate_datasets_losses(
            args.syncnet_checkpoint_path,
            args.syncnet_img_size,
            args.data_root,
            args.cosine_loss_epoch,
        )
        cosine_loss_mean_max = np.quantile(
            [v[0] for v in stat_losses.values()], args.cosine_loss_mean_max_q)
        cosine_loss_std_max = np.quantile(
            [v[1] for v in stat_losses.values()], args.cosine_loss_std_max_q)
        print("filter parameters:")
        print("mean_max_q: {}, mean_max: {}, std_max_q: {}, std_max: {}".format(
            args.cosine_loss_mean_max_q, cosine_loss_mean_max,
            args.cosine_loss_std_max_q, cosine_loss_std_max,
        ))
        for vidname, (mean_loss, std_loss) in stat_losses.items():
            if mean_loss < cosine_loss_mean_max and std_loss < cosine_loss_std_max:
                valid_vidnames.add(vidname)

    if args.filter_outbound_lip:
        land_paths = glob(os.path.join(args.data_root, "**/**/landmarks.npy"))
        path_landmarks = list(filter(lambda item: len(item[1]) > 0, [
            (os.path.dirname(path), np.load(path, allow_pickle=True).tolist()) for path in land_paths]))
        stats_diff = {k: v for k, v in map(
            to_sorted_stats_landmarks, path_landmarks)}
        # {'dirpath': Array(68, [x, y], [mean, std])}

        stats = {k: v for k, v in map(
            partial(to_sorted_stats_landmarks, delta=False), path_landmarks)}
        mid_lip_stats_diff = np.array(list(stats_diff.values()))[:, 52, :, :]
        mid_lip_stats = np.array(list(stats.values()))[:, 52, :, :]
        max_mean_x_diff = np.quantile(
            mid_lip_stats_diff[:, 0, 0], q=args.lip_mean_cut_q)
        max_std_x_diff = np.quantile(
            mid_lip_stats_diff[:, 0, 1], q=args.lip_mean_cut_q)
        min_mean_x = np.quantile(mid_lip_stats[:, 0, 0], q=(
            1.0 - args.lip_mean_cut_q) / 2.0)
        max_mean_x = np.quantile(
            mid_lip_stats[:, 0, 0], q=1.0 - (1.0 - args.lip_mean_cut_q) / 2.0)
        max_std_x = np.quantile(mid_lip_stats[:, 0, 1], q=args.lip_mean_cut_q)
        print("filter_outbound_lip parameter:")
        print(
            "max_mean_x_diff:", max_mean_x_diff,
            "max_std_x_diff", max_std_x_diff,
            "min_mean_x", min_mean_x,
            "max_mean_x", max_mean_x,
            "max_std_x", max_std_x,
        )
    if args.min_mean_blur_score > 0 or args.blur_score_q_cut < 1.0:
        blur_paths = glob(os.path.join(args.data_root, "**/**/blur.npy"))
        blurs_mean = {}
        blurs_std = {}
        for path in blur_paths:
            values = np.load(path, allow_pickle=True).tolist()
            if len(values) == 0:
                continue
            blurs_mean[os.path.dirname(path)] = values
        for dirpath, values in blurs_mean.items():
            lst = list(values.values())
            blurs_mean[dirpath] = np.mean(lst)
            blurs_std[dirpath] = np.std(lst)
        max_std_blur = np.quantile(list(blurs_std.values()), q=args.blur_score_q_cut)
    i_train = 0
    i_val = 0
    train_lines = []
    val_lines = []
    for dirname in os.listdir(args.data_root):
        dirpath = os.path.join(args.data_root, dirname)
        if not os.path.isdir(dirpath):
            continue
        if dirname in include_train_dirs:
            print("force dir {} into training datasets".format(dirname))
            for dataname in tqdm(os.listdir(dirpath), desc="[{}] draw".format(dirname)):
                line = os.path.join(dirname, dataname)
                train_lines.append(line)
            continue
        for dataname in tqdm(os.listdir(dirpath), desc="[{}] draw and filter".format(dirname)):
            dataname_path = os.path.join(dirpath, dataname)
            if args.filter_outbound_lip:
                if dataname_path not in stats_diff:
                    continue
                if stats_diff[dataname_path][52, 0, 0] > max_mean_x_diff:
                    continue
                if stats_diff[dataname_path][52, 0, 1] > max_std_x_diff:
                    continue
                if stats[dataname_path][52, 0, 0] < min_mean_x:
                    continue
                if stats[dataname_path][52, 0, 0] > max_mean_x:
                    continue
                if stats[dataname_path][52, 0, 1] > max_std_x:
                    continue
                if args.min_mean_blur_score > 0 and blurs_mean[dataname_path] < args.min_mean_blur_score:
                    continue
                if args.blur_score_q_cut < 1.0 and blurs_std[dataname_path] > max_std_blur:
                    continue
            line = os.path.join(dirname, dataname)
            if args.syncnet_checkpoint_path is not None:
                if line not in valid_vidnames:
                    continue
            if args.min_img_size > 0:
                paths = []
                for fname in os.listdir(dataname_path):
                    if not fname.endswith('.png'):
                        continue
                    path = os.path.join(dataname_path, fname)
                    paths.append(path)
                samples = np.random.choice(paths, min(len(paths), 64), replace=False)
                sizes = list(map(get_min_img_size, samples))
                min_size = min(sizes)
                if min_size < args.min_img_size:
                    continue
            if np.random.rand() < args.train_ratio:
                train_lines.append(line)
            else:
                val_lines.append(line)
    if len(val_lines) == 0:
        val_lines.append(train_lines.pop())
    np.random.shuffle(train_lines)
    np.random.shuffle(val_lines)

    if not os.path.exists(args.filelists_dir):
        os.makedirs(args.filelists_dir)
    train_path = os.path.join(args.filelists_dir, "train.txt")
    val_path = os.path.join(args.filelists_dir, "val.txt")

    with open(train_path, 'w') as t:
        for i, line in enumerate(train_lines):
            if args.train_limit > 0 and i > args.train_limit:
                break
            t.write(line + "\n")
            i_train += 1

    with open(val_path, 'w') as v:
        for i, line in enumerate(val_lines):
            if args.val_limit > 0 and i > args.val_limit:
                break
            v.write(line + "\n")
            i_val += 1
    print("Create {} train data at: {}".format(i_train, train_path))
    print("Create {} val data at: {}".format(i_val, val_path))


if __name__ == "__main__":
    main()
