from w2l.models.syncnet import SyncNet_color
from w2l.models.wav2lip import Wav2Lip_disc_qual
from w2l.models.blend import LaplacianBlending
from w2l.utils.face_detect import FaceConfigStream, FaceConfigReferenceStream
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
from w2l.utils.loss import cal_blur

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lb = LaplacianBlending().to(device).eval()


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(model_class, path):
    model = model_class()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s, strict=False)

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

    for masked_x, x, mel_batch, frame_batch, coords_batch, masks in stream_loader:
        B = masked_x.size(0)
        mel_batch = torch.reshape(
            mel_batch, [B, 1, hparams.num_mels, hparams.syncnet_mel_step_size])
        half_masks = masks[:, :, hparams.half_img_size:]

        # img_batch: (B, 6, H, W)
        # mouth_mask_batch: (B, 1, H, W)
        # mel_batch: (B, 1, 80, 16)
        # coords_batch: (B, 4)
        yield masked_x, x, half_masks, mel_batch, frame_batch, coords_batch


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
            delta = (np.sqrt(((x ** 2) / (a ** 2)) +
                     ((y ** 2) / (b ** 2))) - 1.) * 2.0
            if delta < 0.:
                continue
            v = min(delta, 1.0)
            filt[iy, ix] = 1.0 - v
            anti_filt[iy, ix] = v
    return filt, anti_filt


def generate_video(face_config_path, audio_path, model_path, output_path, face_fps=25,
                   batch_size=128, num_mels=80, mel_step_size=16, sample_rate=16000,
                   output_fps=None, output_crf=0, start_seconds=0.0):

    # face_filter, anti_face_filter = create_ellipse_filter()
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
    gen = datagen(face_config_path, mel_chunks,
                  batch_size=batch_size, start_frame=start_frame)
    model = load_model(Wav2Lip, model_path)
    print("Model loaded")

    model.eval()

    stream = FaceConfigReferenceStream(face_config_path)
    stream_loader = data_utils.DataLoader(
        stream,
        num_workers=0, batch_size=batch_size)

    reference_embedding = None
    n_frames = stream.video_len

    with torch.no_grad():
        refs = []
        for i, ref in enumerate(tqdm(stream_loader, desc="extract ref embedding")):
            refs.append(ref.unsqueeze(1))
            if i % hparams.syncnet_T == hparams.syncnet_T - 1:
                refs = torch.cat(refs, dim=1)
                B = ref.size(0)

                refs = torch.FloatTensor(refs).permute((0, 1, 4, 2, 3)).to(device) / 255.0
                emb = model.forward_reference(refs) * float(B / n_frames)
                emb = emb.reshape((B, 512, 1, 1)).mean(0, keepdim=True)
                if reference_embedding is None:
                    reference_embedding = emb
                else:
                    reference_embedding += emb
                refs = []

    for i, (masked_img_batch, img_batch, half_masks, mel_batch, frames, coords) in enumerate(
            tqdm(gen, total=len(mel_chunks) // batch_size)):
        if i == 0:
            frame_h, frame_w = frames[0].shape[:-1]
            out = cv2.VideoWriter(
                'temp/result.avi',
                cv2.VideoWriter_fourcc(*'FFV1'), face_fps, (frame_w, frame_h))

        masked_img_batch = masked_img_batch.to(device)
        img_batch = img_batch.to(device)
        mel_batch = mel_batch.to(device)
        half_masks = half_masks.to(device)

        with torch.no_grad():
            # dump_face(img_batch, '/hdd/checkpoints/w2l/temp')
            half_pred = model.inference(
                mel_batch, masked_img_batch, reference_embedding)

            half_pred = lb(img_batch[:, :, hparams.half_img_size:], half_pred, half_masks)
            half_pred = half_pred.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(half_pred, frames, coords):
            f = f.cpu().numpy().astype(np.uint8)
            y1, y2, x1, x2 = c
            face_width = x2 - x1
            face_height = y2 - y1
            half_face_height = face_height // 2
            if face_width > 0 and face_height > 0:
                p = cv2.resize(p, (face_width, half_face_height))
                f[(y2-half_face_height):y2, x1:x2] = p.astype(np.uint8)
            out.write(f)

    out.release()

    command = "ffmpeg -y -i '{}' -i '{}' -vf fps={} -crf {} -vcodec h264 -preset veryslow '{}'".format(
        audio_path, 'temp/result.avi', output_fps, output_crf, output_path)
    subprocess.call(command, shell=platform.system() != 'Windows')


def demo(face_config_path, audio_path, model_path, output_path, disc_path, syncnet_path, face_fps=25,
         batch_size=128, num_mels=80, mel_step_size=16, sample_rate=16000,
         output_fps=None, output_crf=0, start_seconds=0.0):

    assert os.path.exists(disc_path)
    assert os.path.exists(syncnet_path)

    font = cv2.FONT_HERSHEY_SIMPLEX
    topLeftCornerOfText = (50, 50)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    logloss = torch.nn.BCELoss(reduction='none')

    def cosine_loss(a, v, y):
        d = torch.nn.functional.cosine_similarity(a, v)
        loss = logloss(d.unsqueeze(1), y)

        return loss

    def get_sync_loss(syncnet, mel, half_g, expect_true=True):
        B = half_g.size(0)
        # half_g: B x T x 3 x H//2 x W
        half_g = half_g.reshape(
            (B, hparams.syncnet_T * 3, hparams.half_img_size, hparams.img_size))
        # B, T * 3, H//2, W
        a, v = syncnet(mel, half_g)
        if expect_true:
            y = torch.ones((B, 1), dtype=torch.float32, device=device)
        else:
            y = torch.zeros((B, 1), dtype=torch.float32, device=device)
        return cosine_loss(a, v, y).reshape((B,))

    # face_filter, anti_face_filter = create_ellipse_filter()
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
    gen = datagen(face_config_path, mel_chunks,
                  batch_size=batch_size, start_frame=start_frame)
    model = load_model(Wav2Lip, model_path)
    disc = load_model(Wav2Lip_disc_qual, disc_path)
    syncnet = load_model(SyncNet_color, syncnet_path)
    print("Model loaded")

    model.eval()
    disc.eval()
    syncnet.eval()
    real_imgs_for_sync = []
    gen_imgs_for_sync = []
    last_mel = None
    last_gen_syncloss = 0.0
    for i, (img_batch, half_mouth_mask_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=len(mel_chunks) // batch_size)):
        if i == 0:
            frame_h, frame_w = frames[0].shape[:-1]
            out = cv2.VideoWriter(
                'temp/result.avi',
                cv2.VideoWriter_fourcc(*'FFV1'), face_fps, (frame_w * 2, frame_h))

        img_batch = img_batch.to(device)
        mel_batch = mel_batch.to(device)
        half_mouth_mask_batch = half_mouth_mask_batch.to(device)
        half_x = img_batch[:, :, hparams.half_img_size:]
        half_x_truth_batch = half_x[:, :3]
        if last_mel is None:
            last_mel = mel_batch[0:1]

        with torch.no_grad():
            # dump_face(img_batch, '/hdd/checkpoints/w2l/temp')
            half_pred = model(mel_batch, img_batch)

        gen_sync_losses = []
        for i in range(len(half_x)):
            real_imgs_for_sync.append(half_x_truth_batch[i])
            gen_imgs_for_sync.append(half_pred[i])

            if len(real_imgs_for_sync) == hparams.syncnet_T:
                gen_for_sync = torch.unsqueeze(
                    torch.cat(gen_imgs_for_sync, dim=0), 0)
                last_gen_syncloss = get_sync_loss(
                    syncnet, last_mel, gen_for_sync).mean()

                real_imgs_for_sync = []
                gen_imgs_for_sync = []
                last_mel = None

            gen_sync_losses.append(last_gen_syncloss)

        # wrong window is still the real image here
        half_real_img_batch = img_batch[:, 3:, hparams.half_img_size:]
        half_pred = lb(half_real_img_batch, half_pred, half_mouth_mask_batch)
        half_pred = half_pred.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c, gs in zip(
                half_pred, frames, coords, gen_sync_losses):
            f = f.cpu().numpy().astype(np.uint8)
            f_real = f.copy()

            y1, y2, x1, x2 = c
            face_width = x2 - x1
            face_height = y2 - y1
            half_face_height = face_height // 2

            rb = 0.0
            gb = 0.0
            if face_width > 0 and face_height > 0:
                p = cv2.resize(p, (face_width, half_face_height))
                f_of_p = f[(y2-half_face_height):y2, x1:x2].astype(np.float32)
                f[(y2-half_face_height):y2, x1:x2] = p

                rb = cal_blur(f_of_p.astype(np.uint8))
                gb = cal_blur(p.astype(np.uint8))
            loss_info = f'GenSync: {gs:0.3f}, RealBlur: {rb:0.3f}, GenBlur: {gb:0.3f}'
            f_out = np.concatenate((f_real, f), axis=1)
            cv2.putText(
                f_out, loss_info,
                topLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
            out.write(f_out)

    out.release()

    command = "ffmpeg -y -i '{}' -i '{}' -vf fps={} -crf {} -vcodec h264 -preset veryslow '{}'".format(
        audio_path, 'temp/result.avi', output_fps, output_crf, output_path)
    subprocess.call(command, shell=platform.system() != 'Windows')
