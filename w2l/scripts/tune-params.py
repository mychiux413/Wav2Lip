import sys

sys.path.append('.')
from argparse import Namespace
from w2l.scripts.hq_wav2lip_train import train, eval_model, global_step, global_epoch
from w2l.scripts.hq_wav2lip_train import main as hq_train
import optuna
from w2l.utils.env import use_cuda, device
from w2l.hparams import hparams


def objective(trial):
    global global_step, global_epoch
    global_step = 0
    global_epoch = 0

    target_dir = '/hdd/checkpoints/w2l/exp-{}'.format(trial._trial_id)
    sampling_half_window_size_seconds = trial.suggest_float(
        'sampling_half_window_size_seconds', 1.0, 10.0, log=False)
    img_augment = trial.suggest_categorical('img_augment', [True, False])
    expand_mouth_width_ratio = trial.suggest_float(
        'expand_mouth_width_ratio', 0.3, 0.8, log=False)
    expand_mouth_height_ratio = trial.suggest_float(
        'expand_mouth_height_ratio', 0.3, 0.8, log=False)
    syncnet_lr_range = trial.suggest_float(
        'syncnet_lr_range', 1e-6, 1e-3, log=True)
    syncnet_wt_range = trial.suggest_float(
        'syncnet_wt_range', 0.0, 2.0, log=False)
    disc_wt_range = trial.suggest_float(
        'disc_wt_range', 0.0, 2.0, log=False)
    l1_wt_range = trial.suggest_float(
        'l1_wt_range', 0.0, 2.0, log=False)
    ssim_wt = trial.suggest_float(
        'l1_wt_range', 0.0, 2.0, log=False)
    landmarks_wt = trial.suggest_float(
        'l1_wt_range', 0.0, 10.0, log=False)
    merge_ref_range = trial.suggest_categorical('merge_ref_range', [True, False])
    args = Namespace(
        data_root='datasets/pngs', 
        checkpoint_dir=target_dir,
        syncnet_checkpoint_path='/hdd/checkpoints/w2l/syncnet-mid-210416/checkpoint_step000640000.pth',
        filelists_dir='filelists/filelists-210421',
        train_limit=128,
        val_limit=32,
        checkpoint_path=None,
        disc_checkpoint_path=None,
        )
    hparams.sampling_half_window_size_seconds = sampling_half_window_size_seconds
    hparams.img_augment = img_augment
    hparams.expand_mouth_width_ratio = expand_mouth_width_ratio
    hparams.expand_mouth_height_ratio = expand_mouth_height_ratio
    hparams.syncnet_lr_range = syncnet_lr_range
    hparams.syncnet_wt_range = syncnet_wt_range
    hparams.disc_wt_range = disc_wt_range
    hparams.l1_wt_range = l1_wt_range
    hparams.ssim_wt = ssim_wt
    hparams.landmarks_wt = landmarks_wt
    hparams.merge_ref_range = merge_ref_range
    hparams.nepochs = 1
    hq_train(args)
    return 1.0


def main():
    study = optuna.create_study(storage="sqlite:///exp210423.sqlite")  # Create a new study.
    study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.


if __name__ == '__main__':
    main()
