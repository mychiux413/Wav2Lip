import os
import cv2
from w2l.face_detection import FaceAlignment, LandmarksType
from w2l.utils.stream import stream_video_as_batch, get_video_fps_and_frame_count
import pandas as pd
from tqdm import tqdm
import numpy as np
from w2l.hparams import hparams
import torch
from w2l.utils.facenet import load_facenet_model
from w2l.utils.data import cal_mouth_mask_pos


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


def detect_face_and_dump_from_image(img_path, dump_dir, device, face_size, fps=25, pads=None, box=None):
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
        detector = FaceAlignment(
            LandmarksType._2D,
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

    # **** add info info config ****
    raw = open(face_config_path).read()
    with open(face_config_path, 'w') as f:
        f.write('# fps={}\n'.format(fps))
        f.write(raw)
    # ******************************

    return face_config_path


def detect_face_and_dump_from_video(vidpath, dump_dir, device, face_size, face_detect_batch_size=2,
                                    pads=None, box=None, smooth=False, smooth_size=5):
    if pads is None:
        pads = (0, 20, 0, 0)
    if box is None:
        box = (-1, -1, -1, -1)
    os.makedirs(dump_dir, exist_ok=True)
    detector = FaceAlignment(
        LandmarksType._2D,
        flip_input=False, device=device)
    facenet_model = load_facenet_model()

    video_stream = cv2.VideoCapture(vidpath)
    width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_stream.release()

    min_height = 270
    min_width = 480

    resize_ratio = min_width / float(width)
    if height * resize_ratio < min_height:
        resize_ratio = min_height / float(height)
    target_width = int(np.round(width * resize_ratio))
    target_height = int(np.round(height * resize_ratio))

    should_resize = width > target_width or height > target_height
    if not should_resize:
        target_width = width
        target_height = height

    i_image = 0
    rows = []
    pady1, pady2, padx1, padx2 = pads
    _, frame_count = get_video_fps_and_frame_count(vidpath)
    smoothier = None
    mouth_smoothier = None
    x1_edge = hparams.img_size * hparams.x1_mouth_mask_edge
    x2_edge = hparams.img_size * hparams.x2_mouth_mask_edge
    face_size = int(face_size)
    for frames in tqdm(stream_video_as_batch(
            vidpath, face_detect_batch_size, face_detect_batch_size),
            desc="dump face", total=frame_count // face_detect_batch_size):
        # frames have not preprocessed
        if should_resize:
            x = np.array([cv2.resize(frame, (target_width, target_height))
                         for frame in frames], dtype=np.float32)
        else:
            x = np.array(frames, dtype=np.float32)
        # x_for_facenet = np.array(
        #     [cv2.resize(frame, (112, 112))
        #      for frame in frames], dtype=np.float32).transpose((0, 3, 1, 2)) / 255.
        # x_for_facenet = torch.FloatTensor(x_for_facenet).to(device)
        # x_for_facenet: (B, C, 112, 112)
        if box[0] == -1:
            rects = detector.get_detections_for_batch(x)
            cali_rects = []  # (x1, y1, x2, y2)
            face_paths = []
            img_paths = []
            faces = []

            # with torch.no_grad():
            #     landmarks_batch = facenet_model(x_for_facenet)[0]
            #     landmarks_batch = landmarks_batch.reshape(
            #         len(frames), -1, 2).cpu().numpy()
            # landmarks positions in the frames

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
                    y1 = max(0, rect[1])
                    y2 = min(height, rect[3])
                    x1 = max(0, rect[0])
                    x2 = min(width, rect[2])

                    if should_resize:
                        x1 = int(np.round(x1 / resize_ratio))
                        x2 = int(np.round(x2 / resize_ratio))
                        y1 = int(np.round(y1 / resize_ratio))
                        y2 = int(np.round(y2 / resize_ratio))
                    y1 = max(0, y1 - pady1)
                    y2 = min(height, y2 + pady2)
                    x1 = max(0, x1 - padx1)
                    x2 = min(width, x2 + padx2)

                else:
                    x1, x2, y1, y2 = (-1, -1, -1, -1)
                    face_path = None

                if smooth:
                    if smoothier is None:
                        if x1 != -1:
                            smoothier = Smoothier(
                                x1=x1, x2=x2, y1=y1, y2=y2, T=smooth_size)
                            # mouth_smoothier = Smoothier(
                            #     x1=mouth_x1, x2=mouth_x2, y1=mouth_y1, y2=mouth_y2, T=smooth_size)
                    else:
                        x1, x2, y1, y2 = smoothier.smooth(
                            x1=x1, x2=x2, y1=y1, y2=y2)
                        # mouth_x1, mouth_x2, mouth_y1, mouth_y2 = mouth_smoothier.smooth(
                        #     x1=mouth_x1, x2=mouth_x2, y1=mouth_y1, y2=mouth_y2)

                cali_rects.append((x1, y1, x2, y2))
                face_paths.append(face_path)
                img_paths.append(img_path)

                if x1 != -1:
                    face = frame[y1:y2, x1:x2]
                    face = cv2.resize(face, (face_size, face_size))
                    faces.append(face)

                    # Debug
                    # for (_x, _y) in landmarks:
                    #     _x = int(_x * face_size)
                    #     _y = int(_y * face_size)
                    #     face[(_y-2):(_y+2), (_x-2):(_x+2), 1] = 255
                    # ****

                    # cv2.imwrite(face_path, face)
                else:
                    faces.append(
                        np.zeros((face_size, face_size, 3), dtype=np.uint8))
            x_for_facenet = np.array([cv2.resize(face, (112, 112)) for face in faces],
                                     dtype=np.float32).transpose((0, 3, 1, 2)) / 255.
            x_for_facenet = torch.FloatTensor(x_for_facenet).to(device)

            with torch.no_grad():
                landmarks_batch = facenet_model(x_for_facenet)[0]
                landmarks_batch = landmarks_batch.reshape(
                    len(frames), -1, 2).cpu().numpy()
            for img_path, face_path, face, (x1, y1, x2, y2), landmarks in zip(img_paths, face_paths, faces, cali_rects, landmarks_batch):
                # **** mouth *****
                mouth_landmarks = landmarks[48:]
                mouth_x1, mouth_x2, mouth_y1, mouth_y2 = cal_mouth_mask_pos(
                    mouth_landmarks,
                    hparams.img_size,
                    hparams.img_size,
                    x1_edge,
                    x2_edge,
                    )
                if smooth:
                    if mouth_smoothier is None:
                        if x1 != -1:
                            mouth_smoothier = Smoothier(
                                x1=mouth_x1, x2=mouth_x2, y1=mouth_y1, y2=mouth_y2, T=smooth_size)
                    else:
                        mouth_x1, mouth_x2, mouth_y1, mouth_y2 = mouth_smoothier.smooth(
                            x1=mouth_x1, x2=mouth_x2, y1=mouth_y1, y2=mouth_y2)
                # **** mouth ****
                if x1 != -1:
                    # Debug
                    # for (_x, _y) in landmarks:
                    #     _x = int(_x * face_size)
                    #     _y = int(_y * face_size)
                    #     face[(_y-2):(_y+2), (_x-2):(_x+2), 1] = 255
                    # ****
                    cv2.imwrite(face_path, face)

                rows.append({
                    'img_path': img_path,
                    'face_path': face_path,
                    'x1': x1,
                    'x2': x2,
                    'y1': y1,
                    'y2': y2,
                    'mouth_x1': mouth_x1,
                    'mouth_x2': mouth_x2,
                    'mouth_y1': mouth_y1,
                    'mouth_y2': mouth_y2,
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

    # **** add info info config ****
    video_stream = cv2.VideoCapture(vidpath)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    video_stream.release()
    raw = open(face_config_path).read()
    with open(face_config_path, 'w') as f:
        f.write('# fps={}\n'.format(fps))
        f.write(raw)
    # ******************************

    return face_config_path


def stream_from_face_config(config_path, infinite_loop=False, start_frame=0):
    config = pd.read_csv(config_path, sep='\t', comment='#')
    if infinite_loop and len(config) == 1:
        row = config.iloc[0, :]
        img = cv2.imread(row['img_path'])
        face = cv2.imread(row['face_path'])
        x1, x2, y1, y2 = (row['x1'], row['x2'], row['y1'], row['y2'])
        while True:
            yield img, face, (y1, y2, x1, x2)

    assert start_frame < len(config) - 1
    for i, row in config.iterrows():
        if i < start_frame:
            continue
        img = cv2.imread(row['img_path'])
        if not pd.isna(row['face_path']):
            face = cv2.imread(row['face_path'])
        else:
            face = np.zeros(
                (hparams.img_size, hparams.img_size, 3), dtype='uint8')
        x1, x2, y1, y2 = (row['x1'], row['x2'], row['y1'], row['y2'])
        yield img, face, (y1, y2, x1, x2)
    while infinite_loop:
        for i, row in config.iterrows():
            if i < start_frame:
                continue
            img = cv2.imread(row['img_path'])
            if not pd.isna(row['face_path']):
                face = cv2.imread(row['face_path'])
            else:
                face = np.zeros(
                    (hparams.img_size, hparams.img_size, 3), dtype='uint8')
            x1, x2, y1, y2 = (row['x1'], row['x2'], row['y1'], row['y2'])
            yield img, face, (y1, y2, x1, x2)


class FaceConfigStream(object):
    def __init__(self, config_path, mels, start_frame=0):
        self.config_path = config_path
        self.start_frame = start_frame
        self.config = pd.read_csv(config_path, sep='\t', comment='#')
        self.mels = mels.copy()
        self.video_len = len(self.config) - self.start_frame

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        mel = self.mels[idx]
        idx = (idx + self.start_frame) % self.video_len
        row = self.config.iloc[idx]
        img = cv2.imread(row['img_path'])
        if not pd.isna(row['face_path']):
            face = cv2.imread(row['face_path'])
        else:
            face = np.zeros(
                (hparams.img_size, hparams.img_size, 3), dtype='uint8')
        x1, x2, y1, y2 = (row['x1'], row['x2'], row['y1'], row['y2'])
        mouth_x1, mouth_x2, mouth_y1, mouth_y2 = (
            row['mouth_x1'], row['mouth_x2'], row['mouth_y1'], row['mouth_y2'])

        face = torch.FloatTensor(face)
        mel = torch.FloatTensor(mel)
        img = torch.IntTensor(img)
        coords = torch.IntTensor([y1, y2, x1, x2])
        mouths = torch.IntTensor([mouth_x1, mouth_x2, mouth_y1, mouth_y2])
        return face, mel, img, coords, mouths
