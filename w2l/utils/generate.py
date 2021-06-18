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

    max_start_idx = int((len(mel[0]) - mel_step_size) / mel_idx_multiplier)
    for i in range(max_start_idx + 1):
        start_idx = int(i * mel_idx_multiplier)
        end_idx = start_idx + mel_step_size
        chunk = mel[:, start_idx:end_idx]
        mel_chunks.append(chunk)
    return mel_chunks


def dump_face(face_sequences, dump_dir):
    from uuid import uuid4
    import os
    import cv2
    import numpy as np

    print("inference dump face into:", dump_dir)

    B = face_sequences.size(0)
    face_sequences = face_sequences.permute((0, 2, 3, 1))
    real = face_sequences[:, :, :, :3]
    reference = face_sequences[:, :, :, 3:]
    dump = torch.cat([reference, real], dim=2)
    dump = (dump * 255.).detach().cpu().numpy().astype(np.uint8)
    hex = uuid4().hex
    for b in range(B):
        img = dump[b]  # (H, W, 3)
        filename = os.path.join(dump_dir, f'inference_{hex}_{b}.jpg')
        cv2.imwrite(filename, img)


def datagen(config_path, mels, batch_size=128, start_frame=0):
    stream = FaceConfigStream(config_path, mels, start_frame)
    stream_loader = data_utils.DataLoader(
        stream,
        num_workers=0, batch_size=batch_size)
    for img_batch, mel_batch, frame_batch, coords_batch, mouth_batch in stream_loader:
        img_masked = img_batch.clone()
        for j, (x1, x2, y1, y2) in enumerate(mouth_batch):
            img_masked[j, y1:y2, x1:x2] = 0
            # mouth_passer = torch.zeros([hparams.img_size, hparams.img_size, 1], dtype=torch.float32)
            # mouth_passer[y1:y2, x1:x2] = 1
            # img_batch[j] *= mouth_passer

        img_batch = torch.cat((img_masked, img_batch), axis=3) / 255.
        img_batch = img_batch.permute((0, 3, 1, 2))
        mel_batch = torch.reshape(
            mel_batch, [mel_batch.size(0), 1, hparams.num_mels, hparams.syncnet_mel_step_size])

        # img_batch: (B, 6, H, W)
        # mel_batch: (B, 1, 80, 16)
        # coords_batch: (B, 4)
        yield img_batch, mel_batch, frame_batch, coords_batch


def create_ellipse_filter():
    img_size = 640

    width = img_size
    height = img_size // 2

    a = (width // 2) - 90
    b = (height // 2) - 50

    filt = np.ones([height, width, 1], dtype=np.float32)
    anti_filt = np.zeros([height, width, 1], dtype=np.float32)

    for ix in range(width):
        for iy in range(height):
            x = ix - width // 2
            y = iy - height // 2
            delta = (np.sqrt(((x ** 2) / (a ** 2)) + ((y ** 2) / (b ** 2))) - 1.) * 2.0
            if delta < 0.:
                continue
            v = min(delta, 1.0)
            filt[iy, ix] = 1.0 - v
            anti_filt[iy, ix] = v
    return filt, anti_filt


def generate_video(face_config_path, audio_path, model_path, output_path, face_fps=25,
                   batch_size=128, num_mels=80, mel_step_size=16, sample_rate=16000,
                   output_fps=None, output_crf=0, start_seconds=0.0):

    face_filter, anti_face_filter = create_ellipse_filter()
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
    mel_chunks = to_mels(
        audio_path, face_fps,
        num_mels=num_mels, mel_step_size=mel_step_size, sample_rate=sample_rate)
    gen = datagen(face_config_path, mel_chunks, batch_size=batch_size, start_frame=start_frame)
    model = load_model(model_path)
    print("Model loaded")

    helf_img_size = hparams.img_size // 2
    model.eval()
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=len(mel_chunks) // batch_size)):
        if i == 0:
            frame_h, frame_w = frames[0].shape[:-1]
            out = cv2.VideoWriter(
                'temp/result.avi',
                cv2.VideoWriter_fourcc(*'FFV1'), face_fps, (frame_w, frame_h))

        img_batch = img_batch.to(device)
        mel_batch = mel_batch.to(device)

        with torch.no_grad():
            # dump_face(img_batch, '/hdd/checkpoints/w2l/temp')
            pred, _ = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        half_pred = pred[:, helf_img_size:, :, :]

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
