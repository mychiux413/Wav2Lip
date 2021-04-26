from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from w2l.utils import audio

import torch
import numpy as np

from glob import glob

import os
import random
import cv2
from w2l.hparams import hparams
import torchvision

augment = torchvision.transforms.Compose([
    torchvision.transforms.ColorJitter(brightness=(
        0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=0),
])


def get_image_list(data_root, split, limit=0, filelists_dir='filelists'):
    filelist = []
    filelists_path = os.path.join(filelists_dir, "{}.txt".format(split))

    i = 0
    with open(filelists_path) as f:
        for line in f:
            line = line.split('#')[0]
            line = line.strip()
            filelist.append(os.path.join(data_root, line))
            i += 1
            if limit > 0 and i > limit:
                break

    return filelist


class Dataset(object):
    syncnet_T = hparams.syncnet_T
    syncnet_mel_step_size = hparams.syncnet_mel_step_size
    use_landmarks = True

    def __init__(self, split, data_root, inner_shuffle=True,
                 limit=0, sampling_half_window_size_seconds=2.0,
                 img_augment=True,
                 filelists_dir='filelists'):
        self.all_videos = list(filter(
            lambda vidname: os.path.exists(join(vidname, "audio.wav")),
            get_image_list(data_root, split, limit=limit, filelists_dir=filelists_dir)))
        self.img_names = {
            vidname: sorted(
                glob(join(vidname, '*.png')),
                key=lambda name: int(os.path.basename(name).split('.')[0])) for vidname in self.all_videos
        }
        self.landmarks = None
        if self.use_landmarks:
            self.landmarks = {
                vidname: np.load(join(vidname, "landmarks.npy"), allow_pickle=True).tolist() for vidname in self.all_videos
            }

        self.img_size = hparams.img_size
        self.half_img_size = int(self.img_size / 2)
        self.x1_mask_edge = int(self.img_size * 0.25)
        self.x2_mask_edge = int(self.img_size * 0.75)

        self.orig_mels = {}
        for vidname in tqdm(self.all_videos, desc="load mels"):
            mel_path = join(vidname, "mel.npy")
            wavpath = join(vidname, "audio.wav")
            assert os.path.exists(wavpath), wavpath
            if os.path.exists(mel_path):
                try:
                    orig_mel = np.load(mel_path)
                except Exception as err:
                    print(err)
                    wav = audio.load_wav(wavpath, hparams.sample_rate)
                    orig_mel = audio.melspectrogram(wav).T
                    np.save(mel_path, orig_mel)
            else:
                wav = audio.load_wav(wavpath, hparams.sample_rate)
                orig_mel = audio.melspectrogram(wav).T
                np.save(mel_path, orig_mel)
            self.orig_mels[vidname] = orig_mel
        self.data_root = data_root
        self.inner_shuffle = inner_shuffle
        self.all_videos_p = None
        self.linear_space = np.array(range(len(self.all_videos)))
        if inner_shuffle:
            imgs_counts = [len(self.img_names[vidname])
                           for vidname in self.all_videos]
            self.all_videos_p = np.array(imgs_counts) / np.sum(imgs_counts)
        self.sampling_half_window_size_seconds = sampling_half_window_size_seconds
        self.img_augment = img_augment
        if not self.inner_shuffle:
            self.data_len = len(self.all_videos)
        else:
            self.data_len = sum([len(names) for _, names in self.img_names.items()]) // hparams.syncnet_T
        print("data length: ", self.data_len)

    def get_vidname(self, idx):
        if self.inner_shuffle:
            idx = np.random.choice(self.linear_space, p=self.all_videos_p)
        return self.all_videos[idx]

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + self.syncnet_T):
            frame = join(vidname, '{}.png'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None:
            return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (self.img_size, self.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        start_frame_num = self.get_frame_id(
            start_frame) + 1  # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0:
            return None
        for i in range(start_frame_num, start_frame_num + self.syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != self.syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def crop_audio_window(self, spec, start_frame):
        if isinstance(start_frame, int):
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(
                start_frame)  # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + self.syncnet_mel_step_size

        return spec[start_idx: end_idx, :]

    def augment_window(self, tensor):
        # input size: T x 3 x H x W
        return augment(tensor)

    def prepare_window(self, window):
        # output size: 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def mask_window(self, window):
        window[:, :, (window.shape[2]//2):self.fringe_y2,
               self.fringe_x1:self.fringe_x2] = 0.
        return window

    def mask_mouth(self, window, wrong_window, vidname, window_fnames):
        fnames = list(map(os.path.basename, window_fnames))
        landmarks = [self.landmarks[vidname][fname] for fname in fnames]
        masks = []
        for i, landmark in enumerate(landmarks):
            mouth_landmark = landmark[49:]
            mouth_x1 = min(mouth_landmark[:, 0]) * self.img_size
            mouth_x2 = max(mouth_landmark[:, 0]) * self.img_size
            mouth_y1 = min(mouth_landmark[:, 1]) * self.img_size
            mouth_y2 = max(mouth_landmark[:, 1]) * self.img_size
            mouth_width = mouth_x2 - mouth_x1
            mouth_height = mouth_y2 - mouth_y1
            mouth_x1 = max(0, int(mouth_x1 - mouth_width *
                           hparams.expand_mouth_width_ratio - 5))
            mouth_x1 = min(mouth_x1, self.x1_mask_edge)
            mouth_x2 = min(self.img_size, int(
                mouth_x2 + mouth_width * hparams.expand_mouth_width_ratio + 5))
            mouth_x2 = max(mouth_x2, self.x2_mask_edge)
            mouth_y1 = max(self.half_img_size, int(mouth_y1 - mouth_height *
                           hparams.expand_mouth_height_ratio - 5))
            mouth_y2 = min(self.img_size, int(
                mouth_y2 + mouth_height * hparams.expand_mouth_height_ratio + 5))

            mask = None
            mask = np.zeros((1, self.img_size, self.img_size))
            mask[:, mouth_y1:mouth_y2, mouth_x1:mouth_x2] = 1.
            masks.append(mask)
            if wrong_window is not None:
                wrong_window[:, i, :, :] *= mask

            if window is not None:
                window[:, i, mouth_y1:mouth_y2, mouth_x1:mouth_x2] = 0.
        return window, wrong_window, landmarks, masks

    def __len__(self):
        return self.data_len

    def sample_right_wrong_images(self, img_names):
        imgs_len = len(img_names)
        img_idx = random.choice(range(imgs_len))
        img_name = img_names[img_idx]

        min_wrong_idx = max(
            0, int(img_idx - hparams.fps * self.sampling_half_window_size_seconds))
        max_wrong_idx = min(
            imgs_len, int(img_idx + hparams.fps * self.sampling_half_window_size_seconds))
        img_wrong_idx = random.choice(range(min_wrong_idx, max_wrong_idx))
        wrong_img_name = img_names[img_wrong_idx]
        return img_name, wrong_img_name


class Wav2LipDataset(Dataset):
    def __getitem__(self, idx):
        while 1:
            vidname = self.get_vidname(idx)
            img_names = self.img_names[vidname]
            if len(img_names) <= 3 * self.syncnet_T:
                continue

            img_name, wrong_img_name = self.sample_right_wrong_images(
                img_names)

            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            orig_mel = self.orig_mels[vidname]
            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != self.syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None:
                continue

            window = self.prepare_window(window)  # 3 x T x H x W
            y = torch.FloatTensor(window)
            wrong_window = self.prepare_window(wrong_window)  # 3 x T x H x W
            window, wrong_window, landmarks, masks = self.mask_mouth(window, wrong_window, vidname, window_fnames)

            cat = np.concatenate([window, wrong_window, y], axis=1)
            cat = torch.FloatTensor(cat)
            if self.img_augment:
                cat = cat.permute((1, 0, 2, 3))
                cat = self.augment_window(cat)
                cat = cat.permute((1, 0, 2, 3))

            window = cat[:, :self.syncnet_T, :, :]
            wrong_window = cat[:, self.syncnet_T:(self.syncnet_T * 2), :, :]
            y = cat[:, (self.syncnet_T * 2):, :, :]

            if hparams.merge_ref:
                x = window + wrong_window
            else:
                x = torch.cat([window, wrong_window], axis=0)            

            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            landmarks = torch.FloatTensor(landmarks)
            masks = torch.FloatTensor(masks)
            return x, indiv_mels, mel, y, landmarks, masks


class SyncnetDataset(Dataset):
    use_landmarks = False

    def __getitem__(self, idx):
        while 1:
            vidname = self.get_vidname(idx)
            img_names = self.img_names[vidname]
            if len(img_names) <= 3 * self.syncnet_T:
                continue

            img_name, wrong_img_name = self.sample_right_wrong_images(
                img_names)

            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (self.img_size, self.img_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read:
                continue

            orig_mel = self.orig_mels[vidname]
            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != self.syncnet_mel_step_size):
                continue

            x = self.prepare_window(window)  # 3 x T x H x W
            # _, x, landmarks, __ = self.mask_mouth(None, x, vidname, window_fnames)
            x = x[:, :, x.shape[2]//2:]
            x = np.transpose(x, (1, 0, 2, 3))
            x = torch.FloatTensor(x)  # T x 3 x H x W
            if self.img_augment:
                x = self.augment_window(x)
            shape = x.shape
            x = x.reshape((shape[0] * shape[1], shape[2], shape[3]))
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y
