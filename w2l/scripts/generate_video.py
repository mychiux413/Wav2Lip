import os
import argparse
from w2l.hparams import hparams as hp
from w2l.utils import generate_video
import shutil
import subprocess


def main():

    parser = argparse.ArgumentParser(
        description='Inference code to lip-sync videos in the wild using Wav2Lip models')

    parser.add_argument('--checkpoint_path', type=str,
                        help='Name of saved checkpoint to load weights from', required=True)

    parser.add_argument('--face_config_path', type=str,
                        help='Filepath of face config', required=True)
    parser.add_argument('--audio', type=str,
                        help='Filepath of video/audio file to use as raw audio source', required=True)
    parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                        default='results/result_voice.mp4')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size for synthesize', default=64)
    parser.add_argument('--output_fps', type=float,
                        help='Specify output fps', default=None)
    parser.add_argument('--output_crf', type=int,
                        help='Specify output crf', default=0)
    parser.add_argument('--remove_face_dump_dir', action='store_true')
    args = parser.parse_args()

    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = "ffmpeg -y -i '{}' -strict -2 '{}'".format(
            args.audio, 'temp/temp.wav')

        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'
    generate_video(args.face_config_path, args.audio, args.checkpoint_path, args.outfile,
                   batch_size=args.batch_size, num_mels=hp.num_mels,
                   mel_step_size=hp.syncnet_mel_step_size, sample_rate=hp.sample_rate,
                   output_fps=args.output_fps, output_crf=args.output_crf)
    if args.remove_face_dump_dir:
        config_dir = os.path.dirname(args.face_config_path)
        shutil.rmtree(config_dir)


if __name__ == '__main__':
    main()
