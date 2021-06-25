import argparse
import os
import numpy as np
from torch.utils import data as data_utils
from w2l.models.syncnet import SyncNet_color as SyncNet
from w2l.utils.data import Dataset, Mels, sort_to_img_names
import random
import torch
from torch import nn
from w2l.hparams import hparams as hp
from w2l.utils import audio
from tqdm import tqdm
from glob import glob
import cv2
from PIL import Image
from multiprocessing import Pool
from time import time
import pandas as pd


class SyncnetDataset(Dataset):
    def __init__(self, data_root, img_size=96):
        self.all_videos = []
        for dirname in os.listdir(data_root):
            dirpath = os.path.join(data_root, dirname)
            if not os.path.isdir(dirpath):
                continue
            for vid_dirname in tqdm(os.listdir(dirpath), desc="[{}] collect video dirs".format(dirpath)):
                video_path = os.path.join(dirpath, vid_dirname)
                wavpath = os.path.join(video_path, "audio.ogg")
                img_len = len(glob(video_path + "/*.jpg"))
                if img_len < hp.fps * 1.5:
                    print("not enough images: {}".format(dirpath))
                    continue
                if not os.path.exists(wavpath):
                    print("skip missing audio of:", vid_dirname)
                    continue
                self.all_videos.append(video_path)
        self.videos_len = len(self.all_videos)

        self.img_names = {}
        self.orig_mels = Mels()
        with Pool() as p:
            for vidname, imgs in p.imap_unordered(
                    sort_to_img_names,
                    tqdm(self.all_videos, desc="prepare image names"),
                    chunksize=128):
                self.img_names[vidname] = imgs
            for vidname in p.imap_unordered(
                    audio.load_and_dump_mel, tqdm(self.all_videos, desc="load mels"), chunksize=128):
                self.orig_mels[vidname] = None
        self.data_root = data_root
        self.inner_shuffle = False
        self.sampling_half_window_size_seconds = 1e10

        self.img_size = img_size
        self.half_img_size = img_size // 2
        self.data_len = len(self.all_videos)
        print("data length: ", self.data_len)

    def __getitem__(self, idx):
        while 1:
            vidname = self.get_vidname(idx)
            img_names = self.img_names[vidname]
            img_name, wrong_img_name = self.sample_right_wrong_images(
                img_names)

            window_fnames = self.get_window(img_name, vidname)
            if window_fnames is None:
                idx += 1
                continue

            false_window_fnames = self.get_window(wrong_img_name, vidname)
            if false_window_fnames is None:
                idx += 1
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (self.img_size, self.img_size))[
                        self.half_img_size:]
                except Exception as _:
                    all_read = False
                    break

                window.append(img)

            if not all_read:
                idx += 1
                continue

            all_read = True
            for fname in false_window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (self.img_size, self.img_size))[
                        self.half_img_size:]
                except Exception as _:
                    all_read = False
                    break

                window.append(img)

            if not all_read:
                idx += 1
                continue

            orig_mel = self.orig_mels[vidname]
            mel = self.crop_audio_window(orig_mel, img_name)

            if (mel.shape[0] != self.syncnet_mel_step_size):
                idx += 1
                continue

            x = self.prepare_window(window)  # (2T, 3, H, W)
            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            return x, mel, os.path.relpath(vidname, self.data_root)


logloss = nn.BCELoss(reduction='none')


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


def evaluate_datasets_losses(syncnet_checkpoint_path, data_root, epochs=5, only_true_image=False):

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

    device = 'cuda:0'
    sync_model = SyncNet().to(device)

    print("load syncnet checkpoint:" + syncnet_checkpoint_path)
    checkpoint = torch.load(syncnet_checkpoint_path)
    sync_model.load_state_dict(checkpoint["state_dict"])
    test_dataset = SyncnetDataset(
        data_root, img_size=hp.img_size)

    def worker_init_fn(i):
        seed = int(time()) + i * 100
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        return

    data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hp.syncnet_batch_size,
        num_workers=hp.num_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=False)

    sync_model.eval()
    losses = {}

    half_img_size = hp.img_size // 2
    with torch.no_grad():
        for epoch in range(epochs):
            for x, mel, vidnames in tqdm(data_loader, desc="[epoch {}] evaluate sync loss".format(epoch)):
                for vidname in vidnames:
                    if vidname not in losses:
                        losses[vidname] = []

                B = x.size(0)
                if B == 1:
                    continue

                x = x.to(device)
                mel = mel.to(device)
                x_true = x[:, :hp.syncnet_T, :]
                x_false = x[:, hp.syncnet_T:, :]
                x_true = x_true.reshape(
                    (B, hp.syncnet_T * 3, half_img_size, hp.img_size))
                x_false = x_false.reshape(
                    (B, hp.syncnet_T * 3, half_img_size, hp.img_size))

                y_true = torch.ones((B, 1), dtype=torch.float32, device=device)
                y_false = torch.zeros((B, 1), dtype=torch.float32, device=device)

                a, v = sync_model(mel, x_true)
                loss_true = cosine_loss(a, v, y_true)[:, 0].to('cpu').numpy()

                if not only_true_image:
                    a, v = sync_model(mel, x_false)
                    loss_false = cosine_loss(a, v, y_false)[:, 0].to('cpu').numpy()

                loss = loss_true
                if not only_true_image:
                    loss = (loss_true * 0.5 + loss_false * 0.5)
                for vidname, l in zip(vidnames, loss):
                    losses[vidname].append(l)
            print("review losses")
            print(pd.Series([np.mean(loss) for loss in losses.values()]).describe())

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


def to_sorted_stats_landmarks(path, dic):
    values = []
    for _, v in sorted(dic.items(), key=lambda item: int(item[0].split('.')[0])):
        values.append(v)
    rows = np.array(values, dtype=np.float32)
    delta_rows = np.diff(rows, axis=0)
    mean = np.mean(rows, axis=0)
    std = np.std(rows, axis=0)
    delta_mean = np.mean(delta_rows, axis=0)
    delta_std = np.std(delta_rows, axis=0)
    return path, np.stack([mean, std, delta_mean, delta_std], axis=-1)


def stats_landmarks(path):
    dirname = os.path.dirname(path)
    dic = np.load(path, allow_pickle=True).tolist()
    if len(dic) == 0:
        return None, None
    for k, v in dic.items():
        dic[k] = v[48:, 0:1]  # take lip of x only
    return to_sorted_stats_landmarks(dirname, dic)


def stats_blurs(path):
    values = np.load(path, allow_pickle=True).tolist()
    if len(values) == 0:
        return None, None, None
    arr = np.array(list(values.values()), dtype=np.float32)
    return os.path.dirname(path), np.mean(arr), np.std(arr)


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
    parser.add_argument('--cosine_loss_mean_max',
                        help='Pass the loss of datasets under specified mean', default=2.0, type=float)
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
    parser.add_argument('--exclude_train_dirs',
                        help='for exclude dirs into training datasets with comma seperated', default="", type=str)

    args = parser.parse_args()

    assert os.path.exists(args.data_root)
    assert args.train_ratio < 1.0
    assert args.train_ratio > 0.0
    include_train_dirs = list(filter(lambda d: d.rstrip('/'), args.include_train_dirs.split(",")))
    exclude_train_dirs = list(filter(lambda d: d.rstrip('/'), args.exclude_train_dirs.split(",")))

    valid_vidnames = set()
    if args.syncnet_checkpoint_path is not None:
        stat_losses = evaluate_datasets_losses(
            args.syncnet_checkpoint_path,
            args.data_root,
            args.cosine_loss_epoch,
        )
        cosine_loss_mean_max = np.quantile(
            [v[0] for v in stat_losses.values()], args.cosine_loss_mean_max_q)
        cosine_loss_std_max = np.quantile(
            [v[1] for v in stat_losses.values()], args.cosine_loss_std_max_q)
        if args.cosine_loss_mean_max < cosine_loss_mean_max:
            print("rewrite cosine_loss_mean_max from {} to {}".format(
                cosine_loss_mean_max, args.cosine_loss_mean_max,
            ))
            cosine_loss_mean_max = args.cosine_loss_mean_max
        print("filter parameters:")
        print("mean_max_q: {}, mean_max: {}, std_max_q: {}, std_max: {}".format(
            args.cosine_loss_mean_max_q, cosine_loss_mean_max,
            args.cosine_loss_std_max_q, cosine_loss_std_max,
        ))
        for vidname, (mean_loss, std_loss) in stat_losses.items():
            if mean_loss < cosine_loss_mean_max and std_loss < cosine_loss_std_max:
                valid_vidnames.add(vidname)
        print("got valid vidnames:", len(valid_vidnames))
        del stat_losses

    if args.filter_outbound_lip:
        land_paths = glob(os.path.join(args.data_root, "**/**/landmarks.npy"))
        with Pool() as p:
            # {'dirpath': Array(17, [x], [mean, std, delta_mean, delta_std])}
            stats = {k: v for k, v in filter(
                lambda item: item[0] is not None,
                p.imap_unordered(stats_landmarks, tqdm(land_paths, desc="path_landmarks"), chunksize=128))}

            #  mid_lip_stats: [vid_size, 17, 1, 4]
            mid_lip_stats = np.array(list(stats.values()))
            print("mid_lip_stats", mid_lip_stats.shape)
        max_mean_x_diff = np.quantile(
            mid_lip_stats[:, 0, 0, 2], q=args.lip_mean_cut_q)
        max_std_x_diff = np.quantile(
            mid_lip_stats[:, 0, 0, 3], q=args.lip_mean_cut_q)
        min_mean_x = np.quantile(mid_lip_stats[:, 0, 0, 0], q=(
            1.0 - args.lip_mean_cut_q) / 2.0)
        max_mean_x = np.quantile(
            mid_lip_stats[:, 0, 0, 0], q=1.0 - (1.0 - args.lip_mean_cut_q) / 2.0)
        max_std_x = np.quantile(
            mid_lip_stats[:, 0, 0, 1], q=args.lip_mean_cut_q)
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
        with Pool() as p:
            for dirpath, mean, std in p.imap_unordered(
                    stats_blurs, tqdm(blur_paths, desc="load blurs"), chunksize=128):
                if dirpath is None:
                    continue
                blurs_mean[dirpath] = mean
                blurs_std[dirpath] = std
        max_std_blur = np.quantile(
            list(blurs_std.values()), q=args.blur_score_q_cut)
    i_train = 0
    i_val = 0
    train_lines = []
    val_lines = []
    for dirname in os.listdir(args.data_root):
        dirpath = os.path.join(args.data_root, dirname)
        if not os.path.isdir(dirpath):
            continue
        if dirname in exclude_train_dirs:
            continue
        if dirname in include_train_dirs:
            print("force dir {} into training datasets".format(dirname))
            for dataname in tqdm(os.listdir(dirpath), desc="[{}] draw".format(dirname)):
                line = os.path.join(dirname, dataname)
                train_lines.append(line)
            continue
        for dataname in tqdm(os.listdir(dirpath), desc="[{}] draw and filter".format(dirname), maxinterval=128):
            dataname_path = os.path.join(dirpath, dataname)
            if args.filter_outbound_lip:
                if dataname_path not in stats:
                    continue
                if stats[dataname_path][0, 0, 2] > max_mean_x_diff:
                    continue
                if stats[dataname_path][0, 0, 3] > max_std_x_diff:
                    continue
                if stats[dataname_path][0, 0, 0] < min_mean_x:
                    continue
                if stats[dataname_path][0, 0, 0] > max_mean_x:
                    continue
                if stats[dataname_path][0, 0, 1] > max_std_x:
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
                    if not fname.endswith('.jpg'):
                        continue
                    path = os.path.join(dataname_path, fname)
                    paths.append(path)
                samples = np.random.choice(
                    paths, min(len(paths), 64), replace=False)
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
