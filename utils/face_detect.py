import torch
import face_detection
import cv2
import os
from utils.stream import stream_video_as_batch, get_video_fps_and_frame_count
import pandas as pd
from tqdm import tqdm
import numpy as np
from torch.utils import data as data_utils
from hparams import hparams


class Smoothier:
    def __init__(self, x1, x2, y1, y2, T=5):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.T = T
        self.PREV_RATIO = (T - 1.0) / T
        self.AFTER_RATIO = 1.0 / T
    def smooth(self, x1, x2, y1, y2):
        if x1 == -1:
            return x1, x2, y1, y2
        self.x1 = int(self.PREV_RATIO * self.x1 + self.AFTER_RATIO * x1)
        self.x2 = int(self.PREV_RATIO * self.x2 + self.AFTER_RATIO * x2)
        self.y1 = int(self.PREV_RATIO * self.y1 + self.AFTER_RATIO * y1)
        self.y2 = int(self.PREV_RATIO * self.y2 + self.AFTER_RATIO * y2)
        return self.x1, self.x2, self.y1, self.y2

def detect_and_dump_image(img_path, dump_dir, device, face_size, pads=None, box=None):
    if pads is None:
        pads = (0, 0, 0, 0)
    if box is None:
        box = (-1, -1, -1, -1)
    img_path = os.path.join(dump_dir, "img_static.png")
    face_path = os.path.join(dump_dir, "face_static.png")
    frame = cv2.imread(img_path)

    if box[0] != -1:
        pady1, pady2, padx1, padx2 = pads
        os.makedirs(dump_dir, exist_ok=True)
        detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False, device=device)
        rect = detector.get_detections_for_batch(np.array([frame]))[0]
        assert rect is not None
        y1 = max(0, rect[1] - pady1)
        y2 = min(frame.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(frame.shape[1], rect[2] + padx2)
    else:
        y1, y2, x1, x2 = box

    face = frame[y1: y2, x1:x2]
    face = cv2.resize(face, (face_size, face_size))
    cv2.imwrite(face_path, face)
    rows = [{
        'img_path': img_path,
        'face_path': face_path,
        'x1': x1,
        'x2': x2,
        'y1': y1,
        'y2': y2,
    }]
    face_config_path = os.path.join(dump_dir, "face.tsv")
    df = pd.DataFrame(rows)
    df.to_csv(face_config_path, sep='\t', index=None)
    return face_config_path


def detect_and_dump(vidpath, dump_dir, device, face_size, face_detect_batch_size=2,
                    pads=None, box=None, smooth=False, smooth_size=5):
    if pads is None:
        pads = (0, 0, 0, 0)
    if box is None:
        box = (-1, -1, -1, -1)
    os.makedirs(dump_dir, exist_ok=True)
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D,
        flip_input=False, device=device)

    i_image = 0
    rows = []
    pady1, pady2, padx1, padx2 = pads
    _, frame_count = get_video_fps_and_frame_count(vidpath)
    smoothier = None
    for frames in tqdm(stream_video_as_batch(vidpath, face_detect_batch_size, face_detect_batch_size), desc="dump face", total=frame_count // face_detect_batch_size):
        if box[0] == -1:
            rects = detector.get_detections_for_batch(np.array(frames))
            for rect, frame in zip(rects, frames):
                img_path = os.path.join(dump_dir, "img_{}.png".format(
                    i_image
                ))
                cv2.imwrite(img_path, frame)
                face_path = os.path.join(dump_dir, "face_{}.png".format(
                    i_image
                ))
                i_image += 1
                if rect is not None:
                    y1 = max(0, rect[1] - pady1)
                    y2 = min(frame.shape[0], rect[3] + pady2)
                    x1 = max(0, rect[0] - padx1)
                    x2 = min(frame.shape[1], rect[2] + padx2)
                else:
                    x1, x2, y1, y2 = (-1, -1, -1, -1)
                    face_path = None

                if smooth:
                    if smoothier is None:
                        if x1 != -1:
                            smoothier = Smoothier(x1, x2, y1, y2, smooth_size)
                    else:
                        x1, x2, y1, y2 = smoothier.smooth(x1, x2, y1, y2)

                if x1 != -1:
                    face = frame[y1: y2, x1:x2]
                    face = cv2.resize(face, (face_size, face_size))
                    cv2.imwrite(face_path, face)

                rows.append({
                    'img_path': img_path,
                    'face_path': face_path,
                    'x1': x1,
                    'x2': x2,
                    'y1': y1,
                    'y2': y2,
                })
        else:
            for frame in frames:
                img_path = os.path.join(dump_dir, "img_{}.png".format(
                    i_image
                ))
                cv2.imwrite(img_path, frame)
                face_path = os.path.join(dump_dir, "face_{}.png".format(
                    i_image
                ))
                i_image += 1
                y1, y2, x1, x2 = box
                face = frame[y1: y2, x1:x2]
                cv2.imwrite(face_path, face)
                rows.append({
                    'img_path': img_path,
                    'face_path': face_path,
                    'x1': x1,
                    'x2': x2,
                    'y1': y1,
                    'y2': y2,
                })

    face_config_path = os.path.join(dump_dir, "face.tsv")
    df = pd.DataFrame(rows)
    df.to_csv(face_config_path, sep='\t', index=None)
    return face_config_path

def FaceDataset(object):
    def __init__(self, config_path):
        self.config = pd.read_csv(config_path, sep='\t')
        print("{} faces".format(len(self.config)))

    def __len__(self):
        return len(self.config)
    def __getitem__(self, idx):
        row = self.config.iloc[idx]
        img = cv2.imread(row['img_path'])
        if row['face_path']:
            face = cv2.imread(row['face_path'])
        x1, x2, y1, y2 = (row['x1'], row['x2'], row['y1'], row['y2'])
        return img, face, (y1, y2, x1, x2)

def stream_from_face_config(config_path, infinite_loop=False):
    config = pd.read_csv(config_path, sep='\t')
    if infinite_loop and len(config) == 1:
        row = config.iloc[0, :]
        img = cv2.imread(row['img_path'])
        face = cv2.imread(row['face_path'])
        x1, x2, y1, y2 = (row['x1'], row['x2'], row['y1'], row['y2'])
        while True:
            yield img, face, (y1, y2, x1, x2)

    for _, row in config.iterrows():
        img = cv2.imread(row['img_path'])
        face = None
        if row['face_path']:
            face = cv2.imread(row['face_path'])
        x1, x2, y1, y2 = (row['x1'], row['x2'], row['y1'], row['y2'])
        yield img, face, (y1, y2, x1, x2)
    while infinite_loop:
        for _, row in config.iterrows():
            img = cv2.imread(row['img_path'])
            face = None
            if row['face_path']:
                face = cv2.imread(row['face_path'])
            x1, x2, y1, y2 = (row['x1'], row['x2'], row['y1'], row['y2'])
            yield img, face, (y1, y2, x1, x2)
