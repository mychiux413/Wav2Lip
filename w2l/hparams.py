from glob import glob
import os
import json
from distutils.util import strtobool

from numpy.lib.arraysetops import isin


class HParams:
    def __init__(self, **kwargs):
        self.data = {}

        for key, value in kwargs.items():
            self.data[key] = value

        self.overwirte_with_env()

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]

    def set_hparam(self, key, value):
        print("set hparam {} as {}".format(key, value))
        self.data[key] = value

    def overwirte_with_env(self):
        # **** environ control ****
        for key, value in self.data.items():
            env_key = "W2L_" + key.upper()
            value_from_env = os.environ.get(env_key)
            if value_from_env is None:
                continue
            for tp in [bool, float, int, str]:
                if isinstance(value, tp):
                    print("overwrite HParams from environ var: {}={}".format(
                        env_key, value_from_env))
                    if isinstance(value, bool):
                        self.data[key] = strtobool(value_from_env)
                    else:
                        self.data[key] = tp(value_from_env)
                    break
        # *************************

    def to_json(self, path):
        print("dump hparams to: {}".format(path))
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    @classmethod
    def from_json(cls, path):
        print("load hparams from: {}".format(path))
        assert os.path.exists(path)
        with open(path, 'r') as f:
            data = json.load(f)
            obj = cls()
            obj.data = data
            return obj

    def overwrite_by_json(self, path):
        print("overwrite hparams from: {}".format(path))
        assert os.path.exists(path)
        with open(path, 'r') as f:
            data = json.load(f)
        self.data = data


# Default hyperparameters
hparams = HParams(
    num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality
    #  network
    rescale=True,  # Whether to rescale audio prior to preprocessing
    rescaling_max=0.9,  # Rescaling value

    # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
    # It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
    # Does not work if n_ffit is not multiple of hop_size!!
    use_lws=False,

    n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    win_size=800,
    # 16000Hz (corresponding to librispeech) (sox --i <filename>)
    sample_rate=16000,

    # Can replace hop_size parameter. (Recommended: 12.5)
    frame_shift_ms=None,

    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization=True,
    # Whether to normalize mel spectrograms to some predefined range (following below parameters)
    # Only relevant if mel_normalization = True
    allow_clipping_in_normalization=True,
    symmetric_mels=True,
    # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2,
    # faster and cleaner convergence)
    max_abs_value=4.,
    # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not
    # be too big to avoid gradient explosion,
    # not too small for fast convergence)
    # Contribution by @begeekmyfriend
    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude
    # levels. Also allows for better G&L phase reconstruction)
    preemphasize=True,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.

    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To
    # test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    fmax=7600,  # To be increased/reduced depending on data.

    ###################### Our training parameters #################################
    img_size=192,  # 96 or 192
    fps=30,

    batch_size=4,
    initial_learning_rate=0.001,
    learning_rate_decay_rate=0.9,
    min_learning_rate=1e-6,
    opt_amsgrad=True,
    opt_weight_decay=0.0,
    # ctrl + c, stop whenever eval loss is consistently greater than train loss for ~10 epochs
    nepochs=200000000000000000,
    num_workers=2,
    checkpoint_interval=20000,
    eval_interval=20000,
    save_optimizer_state=True,
    warm_up_epochs=5,

    sampling_half_window_size_seconds=2.0,
    img_augment=True,

    # mobilefacenet
    mobilefacenet_model_path='checkpoints/mobilefacenet_model_best.pth.tar',
    expand_mouth_width_ratio=0.6,
    expand_mouth_height_ratio=0.7,

    # is initially zero, will be set automatically to 0.03 later. Leads to faster convergence.
    syncnet_wt=0.005,
    syncnet_batch_size=128,
    syncnet_lr=1e-4,
    syncnet_lr_decay_rate=0.995,
    syncnet_min_lr=1e-7,
    syncnet_eval_interval=20000,
    syncnet_checkpoint_interval=20000,
    syncnet_T=6,
    syncnet_mel_step_size=16,
    syncnet_opt_amsgrad=True,
    syncnet_opt_weight_decay=0.0,

    disc_wt=0.07,
    disc_initial_learning_rate=5e-5,
    disc_learning_rate_decay_rate=0.9,
    disc_min_learning_rate=1e-6,
    disc_opt_amsgrad=True,
    disc_opt_weight_decay=0.0,

    l1_wt=0.5,
    ssim_wt=0.5,
    landmarks_wt=0.0,
    landmarks_points=[2, 5, 8, 11, 14, 31, 33, 35, 48, 51, 54, 57, 62, 66],
)


def hparams_debug_string():
    values = hparams.values()
    hp = ["  %s: %s" % (name, values[name])
          for name in sorted(values) if name != "sentences"]
    return "Hyperparameters:\n" + "\n".join(hp)
