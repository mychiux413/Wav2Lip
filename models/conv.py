import torch
from torch import nn
from torch.nn import functional as F
from hparams import hparams

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


def evaluate_conv_layers(img_size):
    for i in range(100):
        if int(img_size / (2 ** i)) <= 1:
            return i
    raise ValueError("img_size: {}".format(img_size))

def evaluate_new_size_after_conv(input_size, kernel_size, stride, padding):
    padded_size = input_size + 2 * padding
    return int((padded_size - kernel_size) / stride + 1)


def evaluate_new_size_after_transpose_conv(input_size, kernel_size, stride, padding, output_padding=0):
    return stride * (input_size - 1) + kernel_size - 2 * padding + output_padding


def create_audio_encoder(audio_layers, batch_size):
    print("create audio_layers:", audio_layers)
    assert audio_layers <= 7
    sequentials = []
    channels = 16
    mel_channel_size = hparams.num_mels
    mel_step_size = hparams.syncnet_mel_step_size
    BT = batch_size * hparams.syncnet_T
    current_shape = (BT, 1, mel_channel_size, mel_step_size)
    shapes = [current_shape]
    for i in range(audio_layers):
        if i == 0:
            sequentials.append(Conv2d(1, channels * 2, kernel_size=3, stride=1, padding=1))
            sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
            sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))

            mel_channel_size = evaluate_new_size_after_conv(mel_channel_size, 3, 1, 1)
            mel_step_size = evaluate_new_size_after_conv(mel_step_size, 3, 1, 1)
            shapes.append((BT, channels * 2, mel_channel_size, mel_step_size))
        elif i == 1:
            sequentials.append(Conv2d(channels, channels * 2, kernel_size=3, stride=(3, 1), padding=1))
            sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
            sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))

            mel_channel_size = evaluate_new_size_after_conv(mel_channel_size, 3, 3, 1)
            mel_step_size = evaluate_new_size_after_conv(mel_step_size, 3, 1, 1)
            shapes.append((BT, channels * 2, mel_channel_size, mel_step_size))
        elif i == audio_layers - 1:
            sequentials.append(Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=0))
            mel_channel_size = evaluate_new_size_after_conv(mel_channel_size, 3, 1, 0)
            mel_step_size = evaluate_new_size_after_conv(mel_step_size, 3, 1, 0)

            sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=1, stride=1, padding=0))
            mel_channel_size = evaluate_new_size_after_conv(mel_channel_size, 1, 1, 0)
            mel_step_size = evaluate_new_size_after_conv(mel_step_size, 1, 1, 0)
            shapes.append((BT, channels * 2, mel_channel_size, mel_step_size))

        else:
            if audio_layers == 5:
                if i == 2:
                    sequentials.append(Conv2d(channels, channels * 2, kernel_size=3, stride=3, padding=1))
                    sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
                    sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))

                    mel_channel_size = evaluate_new_size_after_conv(mel_channel_size, 3, 3, 1)
                    mel_step_size = evaluate_new_size_after_conv(mel_step_size, 3, 3, 1)
                    shapes.append((BT, channels * 2, mel_channel_size, mel_step_size))

                if i == 3:

                    sequentials.append(Conv2d(channels, channels * 2, kernel_size=3, stride=(3, 2), padding=1))
                    sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
                    sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))

                    mel_channel_size = evaluate_new_size_after_conv(mel_channel_size, 3, 3, 1)
                    mel_step_size = evaluate_new_size_after_conv(mel_step_size, 3, 2, 1)
                    shapes.append((BT, channels * 2, mel_channel_size, mel_step_size))

            elif audio_layers == 6:
                if i == 2:
                    sequentials.append(Conv2d(channels, channels * 2, kernel_size=3, stride=3, padding=1))
                    sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
                    sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))

                    mel_channel_size = evaluate_new_size_after_conv(mel_channel_size, 3, 3, 1)
                    mel_step_size = evaluate_new_size_after_conv(mel_step_size, 3, 3, 1)
                    shapes.append((BT, channels * 2, mel_channel_size, mel_step_size))
                if i == 3:
                    sequentials.append(Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1))
                    sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
                    sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))

                    mel_channel_size = evaluate_new_size_after_conv(mel_channel_size, 3, 2, 1)
                    mel_step_size = evaluate_new_size_after_conv(mel_step_size, 3, 2, 1)
                    shapes.append((BT, channels * 2, mel_channel_size, mel_step_size))
                if i == 4:
                    sequentials.append(Conv2d(channels, channels * 2, kernel_size=3, stride=(2, 1), padding=1))
                    sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
                    sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))

                    mel_channel_size = evaluate_new_size_after_conv(mel_channel_size, 3, 2, 1)
                    mel_step_size = evaluate_new_size_after_conv(mel_step_size, 3, 1, 1)
                    shapes.append((BT, channels * 2, mel_channel_size, mel_step_size))

            elif audio_layers == 7:
                if i == 2:
                    sequentials.append(Conv2d(channels, channels * 2, kernel_size=3, stride=3, padding=1))
                    sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
                    sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))

                    mel_channel_size = evaluate_new_size_after_conv(mel_channel_size, 3, 3, 1)
                    mel_step_size = evaluate_new_size_after_conv(mel_step_size, 3, 3, 1)
                    shapes.append((BT, channels * 2, mel_channel_size, mel_step_size))

                if i == 3:
                    sequentials.append(Conv2d(channels, channels * 2, kernel_size=3, stride=(3, 2), padding=1))
                    sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
                    sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))

                    mel_channel_size = evaluate_new_size_after_conv(mel_channel_size, 3, 3, 1)
                    mel_step_size = evaluate_new_size_after_conv(mel_step_size, 3, 2, 1)
                    shapes.append((BT, channels * 2, mel_channel_size, mel_step_size))

                if i == 4 or i == 5:
                    sequentials.append(Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1))
                    sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
                    sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))

                    mel_channel_size = evaluate_new_size_after_conv(mel_channel_size, 3, 1, 1)
                    mel_step_size = evaluate_new_size_after_conv(mel_step_size, 3, 1, 1)
                    shapes.append((BT, channels * 2, mel_channel_size, mel_step_size))
        channels *= 2
    assert mel_channel_size == 1, mel_channel_size
    assert mel_step_size == 1, mel_step_size

    return nn.Sequential(*sequentials), shapes
