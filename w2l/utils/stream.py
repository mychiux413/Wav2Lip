import cv2
from w2l.hparams import hparams as hp
from w2l.utils import audio
import numpy as np


def stream_mel_chunk(filepath, fps):
    wav = audio.load_wav(filepath, hp.sample_rate)
    mel = audio.melspectrogram(wav)
    print("mel", mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_idx_multiplier = hp.num_mels / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + hp.syncnet_mel_step_size > len(mel[0]):
            yield mel[:, len(mel[0]) - hp.syncnet_mel_step_size:]
            break
        yield mel[:, start_idx: start_idx + hp.syncnet_mel_step_size]
        i += 1


def get_mel_chunks_count(filepath, fps):
    wav = audio.load_wav(filepath, hp.sample_rate)
    mel = audio.melspectrogram(wav)
    mel_idx_multiplier = hp.num_mels / fps
    return int((len(mel[0]) - hp.syncnet_mel_step_size) % mel_idx_multiplier)


def get_video_fps_and_frame_count(filepath):
    video_stream = cv2.VideoCapture(filepath)
    n_frame = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    video_stream.release()
    return fps, n_frame


def stream_video(filepath, infinite_loop=False):
    video_stream = cv2.VideoCapture(filepath)

    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            if infinite_loop:
                video_stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                video_stream.release()
                break
        yield frame


def stream_video_as_batch(filepath, batch_size, steps=1, infinite_loop=False):
    batch = []
    assert steps > 0

    for frame in stream_video(filepath, infinite_loop=infinite_loop):
        if len(batch) == batch_size:
            yield batch
            for _ in range(steps):
                batch.pop(0)
        batch.append(frame)
    if len(batch) > 0:
        yield batch
