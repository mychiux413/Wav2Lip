import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
    raise Exception("Must be using >= Python 3.2")

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse
import os
import cv2
import traceback
import subprocess
from tqdm import tqdm
from glob import glob
from w2l.hparams import hparams as hp

from w2l import face_detection
from w2l.utils.stream import stream_video_as_batch
from w2l.utils.facenet import load_facenet_model
from w2l.utils.env import device
import torch


def process_video_file(vfile, args, gpu_id, fa):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = os.path.join(args.preprocessed_root, dirname, vidname)
    if os.path.exists(fulldir) and len(os.listdir(fulldir)) > 3 * hp.syncnet_T + 2:
        return
    os.makedirs(fulldir, exist_ok=True)

    n_pixels_1080p = 1920 * 1080

    video_stream = cv2.VideoCapture(vfile)
    width = video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)

    n_pixels_of_video = width * height

    video_stream.release()

    if n_pixels_of_video == 0:
        print("invalid video: ", vidname)
        return
    batch_size = max(
        int(n_pixels_1080p / n_pixels_of_video * args.batch_size), 1)

    batches = stream_video_as_batch(vfile, batch_size, steps=batch_size)

    i = -1
    for fb in batches:
        preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):
            i += 1
            if f is None:
                continue

            x1, y1, x2, y2 = f
            height = fb[j].shape[0]
            y2 = min(height, y2 + 20)  # add chin
            cv2.imwrite(os.path.join(fulldir, '{}.png'.format(i)),
                        fb[j][y1:y2, x1:x2])


def process_audio_file(vfile, args, template):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = os.path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    wavpath = os.path.join(fulldir, 'audio.wav')
    if os.path.exists(wavpath):
        return

    command = template.format(vfile, wavpath)
    subprocess.call(command, shell=True)


def process_mouth_position(model, args, vfile):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]
    fulldir = os.path.join(args.preprocessed_root, dirname, vidname)
    config_path = os.path.join(fulldir, "landmarks.npy")
    if os.path.exists(config_path):
        return
    config = {}

    img_batch = []
    imgname_batch = []
    fnames = list(filter(lambda name: name.endswith('.png'), os.listdir(fulldir)))
    fnames_len = len(fnames)
    for i, fname in enumerate(fnames):
        path = os.path.join(fulldir, fname)
        img = cv2.imread(path)
        img = cv2.resize(img, (112, 112))
        img = (img / 255.).transpose((2, 0, 1))
        img_batch.append(img)
        imgname_batch.append(fname)
        if len(img_batch) == args.facenet_batch_size:
            x = np.array(img_batch)
            x = torch.from_numpy(x).float()
            landmarks = model(x.to(device))[0]
            landmarks = landmarks.reshape(len(img_batch), -1, 2).cpu().numpy()
            for j, landmark in enumerate(landmarks):
                config[imgname_batch[j]] = landmark
            img_batch = []
            imgname_batch = []
    if len(img_batch) > 0:
        x = np.array(img_batch)
        x = torch.from_numpy(x).float()
        landmarks = model(x.to(device))[0]
        landmarks = landmarks.reshape(len(img_batch), -1, 2).cpu().numpy()
        for j, landmark in enumerate(landmarks):
            config[imgname_batch[j]] = landmark

    np.save(config_path, config, allow_pickle=True)
    assert len(config) == fnames_len, "dump len vs .png size: {} vs {}".format(
        len(config), fnames_len,
    )


def mp_handler(job):
    vfile, args, gpu_id, fa = job
    try:
        process_video_file(vfile, args, gpu_id, fa)
    except KeyboardInterrupt:
        exit(0)
    except Exception as err:
        print(err)
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
    parser.add_argument(
        '--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
    parser.add_argument(
        "--data_root", help="Root folder of the LRS2 dataset", required=True)
    parser.add_argument("--preprocessed_root",
                        help="Root folder of the preprocessed dataset", required=True)
    parser.add_argument(
        '--facenet_batch_size', help='Batch size of facenet', default=64, type=int)

    args = parser.parse_args()

    fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
                                       device='cuda:{}'.format(id)) for id in range(args.ngpu)]

    template = "ffmpeg -loglevel panic -y -i '{}' -strict -2 '{}'"

    print('Started processing for {} with {} GPUs'.format(
        args.data_root, args.ngpu))

    filelist = glob(os.path.join(args.data_root, '*/*.mp4'))

    jobs = [(vfile, args, i % args.ngpu, fa)
            for i, vfile in enumerate(filelist)]
    p = ThreadPoolExecutor(args.ngpu)
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

    for vfile in tqdm(filelist, desc="dump audio"):
        try:
            process_audio_file(vfile, args, template)
        except KeyboardInterrupt:
            exit(0)
        except Exception as _:  # noqa: F841
            traceback.print_exc()
            continue

    facenet_model = load_facenet_model()
    for vfile in tqdm(filelist, desc="dump landmarks"):
        try:
            with torch.no_grad():
                process_mouth_position(facenet_model, args, vfile)
        except KeyboardInterrupt:
            exit(0)
        except Exception as _:  # noqa: F841
            traceback.print_exc()
            continue


if __name__ == '__main__':
    main()
