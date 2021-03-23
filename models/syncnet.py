import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2d, evaluate_conv_layers, evaluate_new_size_after_conv, create_audio_encoder
from hparams import hparams as hp
import numpy as np

class SyncNet_color(nn.Module):
    def __init__(self):
        super(SyncNet_color, self).__init__()

        n_layers = evaluate_conv_layers(hp.img_size)
        sequentials = []
        channels = 16

        last_face_x_size = hp.img_size // 2
        last_face_y_size = hp.img_size
        for i in range(n_layers):
            if i == 0:
                sequentials.append(Conv2d(3 * hp.syncnet_T, channels * 2, kernel_size=(7, 7), stride=1, padding=3))
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 7, 1, 3)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 7, 1, 3)
            elif i == 1:
                sequentials.append(Conv2d(channels, channels * 2, kernel_size=5, stride=(1, 2), padding=1))
                sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
                sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 5, 1, 1)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 5, 2, 1)
            elif i == n_layers - 1:
                sequentials.append(Conv2d(channels, channels, kernel_size=3, stride=2, padding=1))
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 3, 2, 1)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 3, 2, 1)

                sequentials.append(Conv2d(channels, channels, kernel_size=3, stride=1, padding=0))
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 3, 1, 0)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 3, 1, 0)
    
                sequentials.append(Conv2d(channels, channels, kernel_size=1, stride=1, padding=0))
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 1, 1, 0)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 1, 1, 0)
                
            else:
                sequentials.append(Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1))
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 3, 2, 1)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 3, 2, 1)

                sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
                sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
                if i == 2:
                    sequentials.append(Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
            print("[syncnet] face_encoder x, y", last_face_x_size, last_face_y_size)
            channels *= 2
        assert last_face_x_size == 1, last_face_x_size
        assert last_face_y_size == 1, last_face_y_size
        self.face_encoder = nn.Sequential(*sequentials)

        face_final_channels = channels // 2
        audio_layers = int(np.log(face_final_channels) / np.log(2) - 4)

        self.audio_encoder, audio_shapes = create_audio_encoder(audio_layers, hp.syncnet_batch_size)
        print("[syncnet] review audio_encoder shapes")
        print(*audio_shapes, sep='\n')

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)


        return audio_embedding, face_embedding
