import sys

sys.path.append('.')
from argparse import Namespace
from w2l.scripts.hq_wav2lip_train import train, eval_model, reset_global
from w2l.scripts.hq_wav2lip_train import main as hq_train
import optuna
from w2l.utils.env import use_cuda, device
from w2l.hparams import hparams


def objective(trial):
    target_dir = '/hdd/checkpoints/w2l/experiments/exp-{}'.format(trial._trial_id)
    sampling_half_window_size_seconds = trial.suggest_float(
        'sampling_half_window_size_seconds', 1.0, 10.0, log=False)
    img_augment = trial.suggest_categorical('img_augment', [True, False])
    expand_mouth_width_ratio = trial.suggest_float(
        'expand_mouth_width_ratio', 0.3, 0.8, log=False)
    expand_mouth_height_ratio = trial.suggest_float(
        'expand_mouth_height_ratio', 0.3, 0.8, log=False)
    initial_learning_rate = trial.suggest_float(
        'initial_learning_rate', 1e-7, 1e-3, log=True)
    amsgrad = trial.suggest_categorical('amsgrad', [True, False])
    opt_weight_decay = trial.suggest_categorical('opt_weight_decay', [0.0, 0.01, 0.001])
    syncnet_wt = trial.suggest_float(
        'syncnet_wt', 0.0, 2.0, log=False)
    disc_wt = trial.suggest_float(
        'disc_wt', 0.0, 2.0, log=False)
    l1_wt = trial.suggest_float(
        'l1_wt', 0.0, 2.0, log=False)
    ssim_wt = trial.suggest_float(
        'ssim_wt', 0.0, 2.0, log=False)
    landmarks_wt = trial.suggest_float(
        'landmarks_wt', 0.0, 10.0, log=False)
    merge_ref = trial.suggest_categorical('merge_ref', [True, False])
    args = Namespace(
        data_root='datasets/pngs',
        checkpoint_dir=target_dir,
        syncnet_checkpoint_path='/hdd/checkpoints/w2l/syncnet-mid-210416/checkpoint_step000640000.pth',
        filelists_dir='filelists/filelists-210421',
        train_limit=1280,
        val_limit=256,
        checkpoint_path=None,
        disc_checkpoint_path=None,
        )
    hparams.sampling_half_window_size_seconds = sampling_half_window_size_seconds
    hparams.img_augment = img_augment
    hparams.expand_mouth_width_ratio = expand_mouth_width_ratio
    hparams.expand_mouth_height_ratio = expand_mouth_height_ratio
    hparams.initial_learning_rate = initial_learning_rate
    hparams.disc_initial_learning_rate = initial_learning_rate
    hparams.opt_amsgrad = amsgrad
    hparams.syncnet_opt_amsgrad = amsgrad
    hparams.opt_weight_decay = opt_weight_decay
    hparams.syncnet_opt_weight_decay = opt_weight_decay
    hparams.syncnet_wt = syncnet_wt
    hparams.disc_wt = disc_wt
    hparams.l1_wt = l1_wt
    hparams.ssim_wt = ssim_wt
    hparams.landmarks_wt = landmarks_wt
    hparams.merge_ref = merge_ref
    hparams.nepochs = 1
    reset_global()
    loss = hq_train(args)
    return loss


def main():
    study = optuna.create_study(
        study_name='w2l-experiment',
        storage="sqlite:///exp210426.sqlite",
        load_if_exists=True)  # Create a new study.
    study.optimize(objective, n_trials=500)  # Invoke optimization of the objective function.


if __name__ == '__main__':
    main()
