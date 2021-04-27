import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
    raise Exception("Must be using >= Python 3.2")

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
from w2l.utils.loss import cal_blur
import torch
from multiprocessing import Pool


def process_video_file(fa, vfile, args):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = os.path.join(args.preprocessed_root, dirname, vidname)
    wavpath = os.path.join(fulldir, 'audio.ogg')
    if os.path.exists(wavpath) and len(os.listdir(fulldir)) > 3 * hp.syncnet_T + 2:
        return
    os.makedirs(fulldir, exist_ok=True)

    video_stream = cv2.VideoCapture(vfile)
    width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    min_height = 270
    min_width = 480

    resize_ratio = min_width / float(width)
    if height * resize_ratio < min_height:
        resize_ratio = min_height / float(height)
    target_width = int(np.round(width * resize_ratio))
    target_height = int(np.round(height * resize_ratio))

    should_resize = width > target_width or height > target_height
    n_pixels_of_video = width * height

    video_stream.release()

    if n_pixels_of_video == 0:
        print("invalid video: ", vidname)
        return

    batches = stream_video_as_batch(
        vfile, args.batch_size, steps=args.batch_size)
    resize_factor_height = height / float(target_height)
    resize_factor_width = width / float(target_width)

    i = -1
    for images in batches:
        if should_resize:
            x = [cv2.resize(image, (target_width, target_height))
                 for image in images]
            x = np.array(x)
        else:
            x = np.array(images)
        preds = fa.get_detections_for_batch(x)

        for j, f in enumerate(preds):
            i += 1
            if f is None:
                continue

            x1, y1, x2, y2 = f
            if should_resize:
                x1 = int(np.round(x1 * resize_factor_width))
                x2 = int(np.round(x2 * resize_factor_width))
                y1 = int(np.round(y1 * resize_factor_height))
                y2 = int(np.round(y2 * resize_factor_height))
            y2 = min(height, y2 + 20)  # add chin
            cv2.imwrite(
                os.path.join(fulldir, '{}.jpg'.format(i)),
                images[j][y1:y2, x1:x2], [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )


def process_audio_file(vfile, args, template):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = os.path.join(args.preprocessed_root, dirname, vidname)
    if not os.path.exists(fulldir):
        return

    wavpath = os.path.join(fulldir, 'audio.ogg')
    if os.path.exists(wavpath) and os.stat(wavpath).st_size > 0:
        return

    command = template.format(vfile, wavpath)
    subprocess.call(command, shell=True)


def process_mouth_position(model, args, vfile):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]
    fulldir = os.path.join(args.preprocessed_root, dirname, vidname)
    if not os.path.exists(fulldir):
        return
    config_path = os.path.join(fulldir, "landmarks.npy")
    if os.path.exists(config_path) and os.stat(config_path).st_size > 0:
        return
    config = {}

    img_batch = []
    imgname_batch = []
    fnames = list(
        filter(lambda name: name.endswith('.jpg'), os.listdir(fulldir)))
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
    assert len(config) == fnames_len, "dump len vs .jpg size: {} vs {}".format(
        len(config), fnames_len,
    )


def process_blur_score(job):
    args, vfile = job
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]
    fulldir = os.path.join(args.preprocessed_root, dirname, vidname)
    if not os.path.exists(fulldir):
        return
    config_path = os.path.join(fulldir, "blur.npy")
    if os.path.exists(config_path) and os.stat(config_path).st_size > 0:
        return False
    config = {}
    for fname in filter(lambda name: name.endswith('.jpg'), os.listdir(fulldir)):
        path = os.path.join(fulldir, fname)
        img = cv2.imread(path)
        score = cal_blur(img)
        config[fname] = score
    np.save(config_path, config, allow_pickle=True)
    return True


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
        '--batch_size', help='Single GPU Face detection batch size', default=12, type=int)
    parser.add_argument(
        "--data_root", help="Root folder of the LRS2 dataset", required=True)
    parser.add_argument("--preprocessed_root",
                        help="Root folder of the preprocessed dataset", required=True)
    parser.add_argument(
        '--facenet_batch_size', help='Batch size of facenet', default=64, type=int)
    parser.add_argument(
        '--limit', help='Limit dump files', default=0, type=int)

    args = parser.parse_args()

    fa = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False,
        device='cuda')

    template = "ffmpeg -loglevel panic -y -i '{}' -strict -2 -vn -codec:a libvorbis '{}'"

    print('Started processing from {} to {}'.format(
        args.data_root, args.preprocessed_root))

    filelist = glob(os.path.join(args.data_root, '*/*.mp4'))
    if args.limit > 0:
        print("limit dump files to:", args.limit)
        np.random.seed(1234)
        filelist = np.random.choice(filelist, size=args.limit, replace=False)

    for f in tqdm(filelist, total=len(filelist), desc='dump video'):
        try:
            process_video_file(fa, f, args)
        except KeyboardInterrupt:
            exit(0)
        except Exception as _:  # noqa: F841
            traceback.print_exc()
            continue

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
    del facenet_model

    def gen():
        for vfile in tqdm(filelist, desc="dump blur scores"):
            yield args, vfile

    with Pool(hp.num_workers) as p:
        for _ in p.imap(process_blur_score, gen(), chunksize=20):
            pass


if __name__ == '__main__':
    main()
