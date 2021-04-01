import os
import argparse
from w2l.hparams import hparams as hp
from w2l.utils import detect_face_and_dump_from_image, detect_face_and_dump_from_video
from w2l.utils.env import use_cuda


def main():

    parser = argparse.ArgumentParser(
        description='Inference code to lip-sync videos in the wild using Wav2Lip models')
    parser.add_argument('--face', type=str,
                        help='Filepath of video/image that contains faces to use', required=True)
    parser.add_argument('--static', type=bool,
                        help='If True, then use only first video frame for inference', default=False)
    parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                        default=25., required=False)

    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                        help='Padding (top, bottom, left, right). Please adjust to include chin at least')

    parser.add_argument('--batch_size', type=int,
                        help='Batch size for face detection', default=1)
    parser.add_argument(
        '--box', nargs='+', type=int, default=[-1, -1, -1, -1],
        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
        'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')
    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')
    parser.add_argument('--smooth_size', default=5,
                        type=int, help='Specify the smooth size')
    parser.add_argument('--temp_face_dir', default="/tmp/face_dump", type=str,
                        help='Specify the face dump path of video')

    args = parser.parse_args()

    device = 'cuda' if use_cuda else 'cpu'

    if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        args.static = True

    if not os.path.isfile(args.face):
        raise ValueError(
            '--face argument must be a valid path to video/image file')

    if args.static:
        config_path = detect_face_and_dump_from_image(
            args.face, args.temp_face_dir, device, hp.img_size, fps=args.fps, pads=args.pads, box=args.box)
    else:
        config_path = detect_face_and_dump_from_video(
            args.face, args.temp_face_dir, device, hp.img_size, args.batch_size,
            pads=args.pads, box=args.box, smooth=not args.nosmooth, smooth_size=args.smooth_size)
    print("dump face config at:", config_path)


if __name__ == '__main__':
    main()
