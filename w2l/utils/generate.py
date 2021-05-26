from w2l.utils.face_detect import FaceConfigStream
from w2l.utils import audio
from w2l.hparams import hparams
import numpy as np
from tqdm import tqdm
from w2l.models import Wav2Lip
import torch
import cv2
import subprocess
import platform
import os
from torch.utils import data as data_utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


def to_mels(audio_path, fps, num_mels=80, mel_step_size=16, sample_rate=16000):
    wav = audio.load_wav(audio_path, sample_rate)
    mel = audio.melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = num_mels / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1
    return mel_chunks


def datagen(config_path, mels, batch_size=128, start_frame=0):
    stream = FaceConfigStream(config_path, mels, start_frame)
    stream_loader = data_utils.DataLoader(
        stream,
        num_workers=0, batch_size=batch_size)
    for img_batch, mel_batch, frame_batch, coords_batch, mouth_batch in stream_loader:
        img_masked = img_batch.clone()
        for j, (x1, x2, y1, y2) in enumerate(mouth_batch):
            img_masked[j, y1:y2, x1:x2] = 0
            mouth_passer = np.zeros((hparams.img_size, hparams.img_size, 1), dtype=np.uint8)
            mouth_passer[y1:y2, x1:x2] = 1
            img_batch[j] *= mouth_passer

        img_batch = torch.cat((img_masked, img_batch), axis=3) / 255.
        mel_batch = torch.reshape(
            mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


def generate_video(face_config_path, audio_path, model_path, output_path, face_fps=25,
                   batch_size=128, num_mels=80, mel_step_size=16, sample_rate=16000,
                   output_fps=None, output_crf=0, start_seconds=0.0):

    face_filter = np.load('face-filter.npy')
    anti_face_filter = np.load('face-anti-filter.npy')
    assert os.path.exists(face_config_path)
    with open(face_config_path, 'r') as f:
        firstline = next(f)
        if firstline.startswith('#'):
            splits = firstline.split('fps=')
            if len(splits) > 1:
                face_fps = float(splits[1].strip())
    if output_fps is None:
        output_fps = face_fps

    start_frame = int(np.round(start_seconds * face_fps))
    mels = to_mels(
        audio_path, face_fps,
        num_mels=num_mels, mel_step_size=mel_step_size, sample_rate=sample_rate)
    gen = datagen(face_config_path, mels, batch_size=batch_size, start_frame=start_frame)
    model = load_model(model_path)
    print("Model loaded")
    model.eval()
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=len(mels) // batch_size)):
        if i == 0:
            frame_h, frame_w = frames[0].shape[:-1]
            out = cv2.VideoWriter(
                'temp/result.avi',
                cv2.VideoWriter_fourcc(*'FFV1'), face_fps, (frame_w, frame_h))

        img_batch = img_batch.permute((0, 3, 1, 2)).to(device)
        mel_batch = mel_batch.permute((0, 3, 1, 2)).to(device)

        with torch.no_grad():
            half_pred = model(mel_batch, img_batch)

        half_pred = half_pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(half_pred, frames, coords):
            f = f.cpu().numpy().astype(np.uint8)
            y1, y2, x1, x2 = c
            face_width = x2 - x1
            face_height = y2 - y1
            half_face_height = face_height // 2
            if face_width > 0 and face_height > 0:
                p = cv2.resize(p, (face_width, half_face_height))
                f_of_p = f[(y2-half_face_height):y2, x1:x2].astype(np.float32)
                face_filter = np.expand_dims(cv2.resize(face_filter.copy(), (face_width, half_face_height)), -1)
                anti_face_filter = np.expand_dims(cv2.resize(anti_face_filter.copy(), (face_width, half_face_height)), -1)
                f[(y2-half_face_height):y2, x1:x2] = (face_filter * p + anti_face_filter * f_of_p).astype(np.uint8)
            out.write(f)

    out.release()

    command = "ffmpeg -y -i '{}' -i '{}' -vf fps={} -crf {} -vcodec h264 -preset veryslow '{}'".format(
        audio_path, 'temp/result.avi', output_fps, output_crf, output_path)
    subprocess.call(command, shell=platform.system() != 'Windows')
