import argparse
import os
import cv2
import subprocess
from w2l.hparams import hparams
from tempfile import NamedTemporaryFile
import numpy as np


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dir', type=str)
    parser.add_argument(
        '--output', type=str)

    args = parser.parse_args()
    assert os.path.exists(args.dir)
    jpgs = list(filter(lambda fname: fname.endswith(
        '.jpg'), os.listdir(args.dir)))
    sorted_jpgs = sorted(jpgs, key=lambda k: int(k.rstrip('.jpg')))

    tmp_file = NamedTemporaryFile(suffix='.avi')
    output = cv2.cv2.VideoWriter(
        tmp_file.name,
        cv2.VideoWriter_fourcc(*'FFV1'), hparams.fps, (hparams.img_size, hparams.img_size))

    landmarks_path = os.path.join(args.dir, 'landmarks.npy')
    landmarks = np.load(landmarks_path, allow_pickle=True).tolist()
    for jpg in sorted_jpgs:
        img_path = os.path.join(args.dir, jpg)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (hparams.img_size, hparams.img_size))
        if jpg in landmarks:
            land = landmarks[jpg]
            for (x, y) in land:
                x_pos = int(hparams.img_size * x)
                y_pos = int(hparams.img_size * y)
                y_start = max(0, y_pos - 2)
                y_end = min(hparams.img_size, y_pos + 2)
                x_start = max(0, x_pos - 2)
                x_end = min(hparams.img_size, x_pos + 2)
                img[y_start:y_end, x_start:x_end, 1] = 255
        output.write(img)
    output.release()

    audio_path = os.path.join(args.dir, 'audio.ogg')

    subprocess.run(['ffmpeg', '-i', tmp_file.name,
                   '-i', audio_path, args.output])

    tmp_file.close()


if __name__ == '__main__':
    main()
