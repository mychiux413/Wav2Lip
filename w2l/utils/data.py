from os.path import dirname, join, basename, isfile
from w2l.models import syncnet
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
from multiprocessing import Pool
from collections import defaultdict

augment_for_wav2lip = torchvision.transforms.Compose([
    torchvision.transforms.ColorJitter(brightness=(
        0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0),
])

augment_for_syncnet = torchvision.transforms.Compose([
    torchvision.transforms.ColorJitter(brightness=(
        0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomRotation(5),
])


def sort_to_img_names(vidname):
    imgs = sorted(glob(os.path.join(vidname, '*.jpg')),
                  key=lambda name: int(os.path.basename(name).split('.')[0]))
    return vidname, imgs


def get_image_list(data_root, split, limit=0, filelists_dir='filelists'):
    filelist = []
    filelists_path = os.path.join(filelists_dir, "{}.txt".format(split))
    assert os.path.exists(filelists_path)

    i = 0
    with open(filelists_path) as f:
        for line in tqdm(f, "scan data root"):
            line = line.split('#')[0]
            line = line.strip()
            dirpath = os.path.join(data_root, line)
            if not os.path.exists(dirpath):
                print("dirpath not exists: {}".format(dirpath))
                continue
            audio_path = os.path.join(dirpath, "audio.ogg")
            if not os.path.exists(audio_path):
                print("audio not exists: {}".format(audio_path))
                continue
            img_len = len(glob(dirpath + "/*.jpg"))
            if img_len < hparams.fps * 1.5:
                print("not enough images: {}".format(dirpath))
                continue
            filelist.append(dirpath)
            i += 1
            if limit > 0 and i > limit:
                break
    return filelist


class Mels(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)

    def __setitem__(self, key, value):
        mel_path = join(key, 'mel.npy')
        assert os.path.exists(mel_path)
        self.__dict__[key] = mel_path

    def __getitem__(self, key):
        mel_path = self.__dict__[key]
        return np.load(mel_path)


class LandMarks(dict):
    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        path = self.__dict__[key]
        return np.load(path, allow_pickle=True).tolist()


class Blurs(dict):
    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        path = self.__dict__[key]
        return np.load(path, allow_pickle=True).tolist()


def cal_mouth_mask_pos(landmarks, img_height, img_width):
    # mouth_landmarks = landmarks[48:]
    x1_mask_edge = landmarks[3, 0]
    x2_mask_edge = landmarks[13, 0]
    y_nose = landmarks[33, 1]
    y_chin = landmarks[8, 1]

    x1_mask_edge = int(x1_mask_edge * img_width + 5)
    x2_mask_edge = int(x2_mask_edge * img_width - 5)
    y1_mask_edge = int(y_nose * img_height)
    y2_mask_edge = int(y_chin * img_height - 10)

    mouth_x1, mouth_x2, mouth_y1, mouth_y2 = x1_mask_edge, x2_mask_edge, y1_mask_edge, y2_mask_edge

    mouth_x1 = max(0, mouth_x1)
    mouth_x2 = min(img_width, mouth_x2)
    mouth_y1 = max(img_height // 2, mouth_y1)
    mouth_y2 = min(img_height, mouth_y2)

    return mouth_x1, mouth_x2, mouth_y1, mouth_y2

# def cal_mouth_contour_mask(landmarks, img_height, img_width):
#     # mouth_landmarks = landmarks[48:]
#     delta_face_width = (landmarks[14, 0] - landmarks[2, 0]) * 0.05
#     delta_face_height = (landmarks[33, 1] - landmarks[8, 1]) * 0.1
#     mouth_contours = [
#         [landmarks[2, 0] + delta_face_width, landmarks[2, 1]],
#         [landmarks[5, 0] + delta_face_width, landmarks[2, 1]],
#         [landmarks[8, 0], landmarks[2, 1] - delta_face_height],
#         [landmarks[11, 0] - delta_face_width, landmarks[2, 1]],
#         [landmarks[14, 0] - delta_face_width, landmarks[2, 1]],
#         [landmarks[33, 0], landmarks[2, 1] + delta_face_height],
#     ]
#     mouth_contours = np.array(mouth_contours, dtype=np.float)
#     mouth_contours[:, 1] = np.maximum(mouth_contours[:, 1], 0.5)

#     return mouth_contours

class Dataset(object):
    valid_sampling_width = hparams.fps + 1
    syncnet_T = hparams.syncnet_T
    syncnet_mel_step_size = hparams.syncnet_mel_step_size
    use_landmarks = True
    use_blurs = True

    def __init__(self, split, data_root, inner_shuffle=True,
                 limit=0, sampling_half_window_size_seconds=2.0,
                 img_augment=True,
                 filelists_dir='filelists',
                 use_syncnet_weights=True):
        self.all_videos = list(filter(
            lambda vidname: os.path.exists(join(vidname, "audio.ogg")),
            get_image_list(data_root, split, limit=limit, filelists_dir=filelists_dir)))
        assert len(self.all_videos) > 0, "no video dirs found from: {} with filelists_dir: {}".format(
            data_root, filelists_dir,
        )
        self.synclosses = {}
        synclosses_path = os.path.join(data_root, "synclosses.npy")
        if os.path.exists(synclosses_path) and use_syncnet_weights:
            self.synclosses = np.load(synclosses_path, allow_pickle=True).tolist()
            videos_set = set(self.all_videos)
            drops = []

            keys = list(self.synclosses.keys())
            for key in keys:
                self.synclosses[os.path.join(data_root, key)] = self.synclosses.pop(key)

            for vidname in self.synclosses.keys():
                if vidname not in videos_set:
                    drops.append(vidname)

            for drop in drops:
                self.synclosses.pop(drop)
            for vidname in self.all_videos:
                if vidname not in self.synclosses:
                    print("vidname not found in synclosses: {}".format(vidname))
                    continue
                self.synclosses[vidname] = np.mean(self.synclosses[vidname])

            means = list(self.synclosses.values())
            min_mean_syncloss = min(means)
            max_mean_syncloss = max(means)
            width = max_mean_syncloss - min_mean_syncloss
            for vidname in self.synclosses.keys():
                self.synclosses[vidname] = max(0.0, (self.synclosses[vidname] - max_mean_syncloss) * -1.0 / width)

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
        self.landmarks = LandMarks()
        if self.use_landmarks:
            print("load landmarks")
            for vidname in self.all_videos:
                self.landmarks[vidname] = join(vidname, "landmarks.npy")
        for vidname in self.landmarks.keys():
            assert vidname in self.orig_mels, "vidname {} is in landmarks but not in orig_mels".format(
                vidname)

        self.blurs = Blurs()
        if self.use_blurs:
            print("load blurs")
            for vidname in self.all_videos:
                self.blurs[vidname] = join(vidname, "blur.npy")
        for vidname in self.blurs.keys():
            assert vidname in self.orig_mels, "vidname {} is in blurs but not in orig_mels".format(
                vidname)

        self.img_size = hparams.img_size
        self.half_img_size = int(self.img_size / 2)
        self.x1_mask_edge = int(self.img_size * hparams.x1_mouth_mask_edge)
        self.x2_mask_edge = int(self.img_size * hparams.x2_mouth_mask_edge)
        self.data_root = data_root
        self.inner_shuffle = inner_shuffle
        self.linear_space = np.array(range(len(self.all_videos)))
        self.sampling_half_window_size_seconds = sampling_half_window_size_seconds
        self.img_augment = img_augment
        self.data_len = len(self.all_videos)
        self.videos_len = len(self.all_videos)
        if self.data_len < 10000:
            self.data_len = min(10000, int(
                sum([len(self.img_names[v]) for v in self.all_videos]) / self.syncnet_T))
        print("data length: ", self.data_len)

    def get_data_weight(self, vidname):
        return self.synclosses.get(vidname, 1.0)

    def get_vidname(self, idx):
        if self.inner_shuffle:
            idx = np.random.choice(self.linear_space)
        return self.all_videos[idx % self.videos_len]

    def get_frame_id(self, frame):
        # 0.jpg is the first image
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame, vidname):
        start_id = self.get_frame_id(start_frame)

        window_fnames = []
        window_base_fnames = []
        for frame_id in range(start_id, start_id + self.syncnet_T):
            fname = '{}.jpg'.format(frame_id)
            frame = join(vidname, fname)
            if not isfile(frame):
                return None, None
            window_fnames.append(frame)
            window_base_fnames.append(fname)
        return window_fnames, window_base_fnames

    def read_window(self, window_fnames):
        # output: [(H, W, 3), ...]
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
        return augment_for_wav2lip(tensor)

    def prepare_window(self, window):
        # window: [(H, W, 3), ...]

        # (2*T, H, W, 3)
        x = np.asarray(window) / 255.

        # (2*T, 3, H, W)
        x = np.transpose(x, (0, 3, 1, 2))

        # output size: (2*T, 3, H, W)
        return x

    def mask_window(self, window):
        window[:, :, (window.shape[2]//2):self.fringe_y2,
               self.fringe_x1:self.fringe_x2] = 0.
        return window

    def get_blurs(self, vidname, window_base_fnames):
        return [self.blurs[vidname][fname] for fname in window_base_fnames]

    def mask_mouth(self, window, vidname, window_base_fnames):
        landmarks = [self.landmarks[vidname][fname] for fname in window_base_fnames]
        # masks = []
        for i, landmark in enumerate(landmarks):
            mouth_x1, mouth_x2, mouth_y1, mouth_y2 = cal_mouth_mask_pos(
                landmark,
                self.img_size,
                self.img_size)

            if window is not None:
                window[i, :, mouth_y1:mouth_y2, mouth_x1:mouth_x2] = 0.

        target_landmarks = [landmark[hparams.landmarks_points]
                            for landmark in landmarks]
        return window, target_landmarks

    def __len__(self):
        return self.data_len

    def sample_right_wrong_images(self, img_names):
        imgs_len = len(img_names)
        img_idx = random.choice(range(imgs_len))
        img_name = img_names[img_idx]

        goleft = random.choice([True, False])
        if img_idx < self.valid_sampling_width:
            goleft = False
        if img_idx > imgs_len - self.valid_sampling_width:
            goleft = True

        if goleft:
            min_wrong_idx = max(
                0, int(img_idx - hparams.fps * self.sampling_half_window_size_seconds))
            max_wrong_idx = img_idx
        else:
            min_wrong_idx = int(img_idx + 1)
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

            img_name, wrong_img_name = self.sample_right_wrong_images(
                img_names)

            window_fnames, window_base_fnames = self.get_window(img_name, vidname)
            wrong_window_fnames, _ = self.get_window(wrong_img_name, vidname)
            if window_fnames is None or wrong_window_fnames is None:
                idx += 1
                continue

            window = self.read_window(window_fnames)
            if window is None:
                idx += 1
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                idx += 1
                continue

            orig_mel = self.orig_mels[vidname]
            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != self.syncnet_mel_step_size):
                idx += 1
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None:
                idx += 1
                continue

            window = self.prepare_window(window)  # T x 3 x H x W
            y = torch.FloatTensor(window)
            wrong_window = self.prepare_window(wrong_window)  # T x 3 x H x W

            cat = np.concatenate([window, wrong_window, y], axis=0)
            cat = torch.FloatTensor(cat)
            if self.img_augment:
                cat = self.augment_window(cat)

            window = cat[:self.syncnet_T, :, :, :]
            wrong_window = cat[self.syncnet_T:(self.syncnet_T * 2):, :, :]
            y = cat[(self.syncnet_T * 2):, :, :, :]

            window, landmarks = self.mask_mouth(
                window, vidname, window_base_fnames)

            x = torch.cat([window, wrong_window], axis=1)

            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            landmarks = torch.FloatTensor(landmarks)

            blurs = self.get_blurs(vidname, window_base_fnames)
            blurs = torch.FloatTensor(blurs)

            data_weight = self.get_data_weight(vidname)

            # x: (T, 6, H, W)
            # y: (T, 3, H, W)
            # indiv_mels: (T, 1, 80, 16)
            # mel: (1, 80, 16)
            return x, indiv_mels, mel, y, landmarks, blurs, data_weight


class SyncnetDataset(Dataset):
    use_landmarks = False

    def augment_window(self, tensor):
        # input size: T x 3 x H x W
        return augment_for_syncnet(tensor)

    def __getitem__(self, idx):
        while 1:
            vidname = self.get_vidname(idx)
            img_names = self.img_names[vidname]
            img_name, wrong_img_name = self.sample_right_wrong_images(
                img_names)

            window_fnames, _ = self.get_window(img_name, vidname)
            if window_fnames is None:
                idx += 1
                continue

            false_window_fnames, _ = self.get_window(wrong_img_name, vidname)
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

            # window: [(H, W, 3), ...]

            orig_mel = self.orig_mels[vidname]
            mel = self.crop_audio_window(orig_mel, img_name)

            # dump_as_video(window_fnames, vidname, self.get_frame_id(img_name), orig_mel)

            if (mel.shape[0] != self.syncnet_mel_step_size):
                idx += 1
                continue

            x = self.prepare_window(window)  # (2T, 3, H, W)
            x = torch.FloatTensor(x)
            if self.img_augment:
                x = self.augment_window(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            data_weight = self.get_data_weight(vidname)

            return x, mel, data_weight


# def dump_as_video(window_fnames, vidname, start_frame_num, spec):
#     from pydub import AudioSegment
#     start_idx = int(80. * (start_frame_num / float(hparams.fps)))
#     end_idx = start_idx + hparams.syncnet_mel_step_size
#     spec_len = len(spec)
#     print("window_fnames", window_fnames)
#     print("vidname", vidname)
#     audio_path = os.path.join(vidname, "audio.ogg")
#     sound = AudioSegment.from_file(audio_path)
#     sound_len = len(sound)
#     chunk_start = int(sound_len * start_idx / spec_len)
#     chunk_end = int(sound_len * end_idx / spec_len)
#     chunk = sound[chunk_start:chunk_end]
#     chunk.export('')
