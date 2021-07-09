from functools import partial
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
from w2l.utils.face_detect import Smoothier, square_positions
from w2l.utils.stream import stream_video_as_batch
from w2l.utils.facenet import load_facenet_model
from w2l.utils.env import device
from w2l.utils.loss import cal_blur
import torch
from multiprocessing import Pool
from pydub import AudioSegment
from shutil import move, rmtree


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

    smoothier = None
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

            if smoothier is None:
                smoothier = Smoothier(x1, x2, y1, y2, T=hp.syncnet_T)
            else:
                x1, x2, y1, y2 = smoothier.smooth(x1, x2, y1, y2)

            x1, x2, y1, y2 = square_positions(x1, x2, y1, y2)

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
        half_img = img[img.shape[0] // 2:]
        score = cal_blur(half_img)
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


def jpg_to_int(filename):
    return int(filename.rstrip('.jpg'))


def move_jpg(from_dir, filename, to_dir, new_filename):
    from_path = os.path.join(from_dir, filename)
    to_path = os.path.join(to_dir, new_filename)
    move(from_path, to_path)


def split_dir(dirpath, max_seconds=20.0):
    if dirpath.endswith('/'):
        dirpath = dirpath[:-1]
    if not os.path.exists(dirpath):
        print("dirpath not exists: {}".format(dirpath))
        return
    audio_path = os.path.join(dirpath, 'audio.ogg')
    if not os.path.exists(audio_path):
        print("missing audio: {}".format(audio_path))
        return
    audio = AudioSegment.from_file(audio_path)
    audio_len = len(audio)
    n_split = np.math.ceil(audio.duration_seconds / max_seconds)
    if n_split < 2:
        return

    jpgs = list(filter(lambda path: path.endswith('.jpg'), os.listdir(dirpath)))
    blur_path = os.path.join(dirpath, 'blur.npy')
    landmarks_path = os.path.join(dirpath, 'landmarks.npy')
    blur = None
    landmarks = None
    if os.path.exists(blur_path):
        blur = np.load(blur_path, allow_pickle=True).tolist()
    if os.path.exists(landmarks_path):
        landmarks = np.load(landmarks_path, allow_pickle=True).tolist()
    for i in range(n_split):
        subdirpath = '{}___{}'.format(dirpath, i)
        if os.path.exists(subdirpath):
            print('redo subdir:', subdirpath)
            rmtree(subdirpath)

        sub_min_frames = int(i * max_seconds * hp.fps)
        sub_max_frames = int((i + 1) * max_seconds * hp.fps)
        sub_jpgs = set(filter(lambda jpg: sub_min_frames <=
                       jpg_to_int(jpg) < sub_max_frames, jpgs))
        if len(sub_jpgs) < int(3 * hp.fps):
            print("skip short split: {}".format(len(sub_jpgs)))
            continue
        print("create subdir: ", subdirpath)
        os.makedirs(subdirpath, exist_ok=True)
        sub_jpgs_map = {}
        for jpg in sub_jpgs:
            current_frame = jpg_to_int(jpg)
            new_jpg = '{}.jpg'.format(current_frame - sub_min_frames)
            sub_jpgs_map[jpg] = new_jpg
        for jpg in sub_jpgs:
            current_frame = jpg_to_int(jpg)
            move_jpg(dirpath, jpg, subdirpath, sub_jpgs_map[jpg])
        if blur is not None:
            sub_blur = {}
            for from_jpg, to_jpg in sub_jpgs_map.items():
                if from_jpg in blur:
                    sub_blur[to_jpg] = blur[from_jpg]
            sub_blur_path = os.path.join(subdirpath, 'blur.npy')
            np.save(sub_blur_path, sub_blur, allow_pickle=True)
        if landmarks is not None:
            sub_landmarks = {}
            for from_jpg, to_jpg in sub_jpgs_map.items():
                if from_jpg in landmarks:
                    sub_landmarks[to_jpg] = landmarks[from_jpg]
            sub_landmarks_path = os.path.join(subdirpath, 'landmarks.npy')
            np.save(sub_landmarks_path, sub_landmarks, allow_pickle=True)
        start_audio_index = int((i * max_seconds) * 1000)
        end_audio_index = min(audio_len, int((i + 1) * max_seconds * 1000))
        sub_audio = audio[start_audio_index:end_audio_index]
        sub_audio.export(os.path.join(subdirpath, 'audio.ogg'), format='ogg')

    print("remove dir", dirpath)
    rmtree(dirpath)


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
    parser.add_argument(
        '--excludes', help='exclude dirs with comma separated', default=None, type=str)
    parser.add_argument(
        '--includes', help='include dirs with comma separated', default=None, type=str)
    parser.add_argument(
        '--max_video_duration_seconds', help='to split every video length under the specified value', default=20.0, type=float)

    args = parser.parse_args()

    fa = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False,
        device='cuda')

    template = "ffmpeg -loglevel panic -y -i '{}' -strict -2 -vn -codec:a libvorbis '{}'"

    print('Started processing from {} to {}'.format(
        args.data_root, args.preprocessed_root))

    filelist = glob(os.path.join(args.data_root, '*/*.mp4'))
    exclude_dirs = tuple()
    if args.excludes is not None:
        excludes = args.excludes.split(',')
        exclude_dirs = tuple(
            set([os.path.join(args.data_root, ex) for ex in excludes]))
    if args.limit > 0:
        print("limit dump files to:", args.limit)
        np.random.seed(1234)
        filelist = np.random.choice(filelist, size=args.limit, replace=False)
    if args.includes is not None:
        includes = args.includes.split(',')
        include_dirs = tuple(
            set([os.path.join(args.data_root, inc) for inc in includes]))
        include_list = glob(os.path.join(include_dirs, '*/*.mp4'))
        filelist = set(filelist + include_list)

    # for f in tqdm(filelist, total=len(filelist), desc='dump video'):
    #     if f.startswith(exclude_dirs):
    #         continue
    #     try:
    #         process_video_file(fa, f, args)
    #     except KeyboardInterrupt:
    #         exit(0)
    #     except Exception as _:  # noqa: F841
    #         traceback.print_exc()
    #         continue

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

    finished_dirs = []
    for datasetname in os.listdir(args.preprocessed_root):
        if args.excludes is not None:
            if datasetname in excludes:
                print("skip to split exclude dir:", datasetname)
                continue
        dirpath = os.path.join(args.preprocessed_root, datasetname)
        if not os.path.isdir(dirpath):
            continue
        for vidname in os.listdir(dirpath):
            if '___' in vidname:
                continue
            vid_dir = os.path.join(dirpath, vidname)
            finished_dirs.append(vid_dir)

    split_func = partial(split_dir, max_seconds=args.max_video_duration_seconds)
    with Pool(hp.num_workers) as p:
        p.map(split_func, finished_dirs)


if __name__ == '__main__':
    main()
