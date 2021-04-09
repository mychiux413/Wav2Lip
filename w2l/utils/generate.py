from w2l.utils.face_detect import stream_from_face_config
from w2l.utils import audio
import numpy as np
from tqdm import tqdm
from w2l.models import Wav2Lip
import torch
import cv2
import subprocess
import platform
import os

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


def datagen(config_path, mels, batch_size=128):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    stream = stream_from_face_config(config_path, infinite_loop=True)

    for i, m in enumerate(mels):
        frame_to_save, face, coords = next(stream)
        if i == 0:
            img_size = face.shape[0]
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(
            mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


def generate_video(face_config_path, audio_path, model_path, output_path, face_fps=25,
                   batch_size=128, num_mels=80, mel_step_size=16, sample_rate=16000,
                   output_fps=30, output_crf=0):

    assert os.path.exists(face_config_path)
    with open(face_config_path, 'r') as f:
        firstline = next(f)
        if firstline.startswith('#'):
            splits = firstline.split('fps=')
            if len(splits) > 1:
                face_fps = int(splits[1].strip())

    mels = to_mels(
        audio_path, face_fps,
        num_mels=num_mels, mel_step_size=mel_step_size, sample_rate=sample_rate)
    gen = datagen(face_config_path, mels, batch_size=batch_size)
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=len(mels) // batch_size)):
        if i == 0:
            model = load_model(model_path)
            print("Model loaded")

            frame_h, frame_w = frames[0].shape[:-1]
            out = cv2.VideoWriter(
                'temp/result.avi',
                cv2.VideoWriter_fourcc(*'FFV1'), face_fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(
            np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(
            np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()

    command = "ffmpeg -y -i '{}' -i '{}' -vf fps={} -crf {} -vcodec h264 -preset veryslow '{}'".format(
        audio_path, 'temp/result.avi', output_fps, output_crf, output_path)
    subprocess.call(command, shell=platform.system() != 'Windows')
