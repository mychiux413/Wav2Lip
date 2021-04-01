import torch
from torch import nn
from torch.nn import functional as F
import math
from w2l.hparams import hparams as hp
import numpy as np

from w2l.models.conv import Conv2dTranspose, Conv2d, nonorm_Conv2d, evaluate_conv_layers, evaluate_new_size_after_conv, create_audio_encoder
from w2l.models.conv import evaluate_new_size_after_transpose_conv

class Wav2Lip(nn.Module):
    def __init__(self):
        super(Wav2Lip, self).__init__()

        n_layers = evaluate_conv_layers(hp.img_size) + 1
        sequentials = []
        channels = 8

        last_face_x_size = hp.img_size
        last_face_y_size = hp.img_size
        face_encoder_channels = [6]

        print("n_layers", n_layers)
        for i in range(n_layers):
            if i == 0:
                sequentials.append(
                    nn.Sequential(Conv2d(6, channels * 2, kernel_size=7, stride=1, padding=3))
                )
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 7, 1, 3)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 7, 1, 3)
                face_encoder_channels.append(channels * 2)
            elif i == 2:
                sequentials.append(
                    nn.Sequential(
                        Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
                        Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True),
                        Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True),
                        Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
                )
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 3, 2, 1)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 3, 2, 1)
                face_encoder_channels.append(channels * 2)
            elif i == n_layers - 2:
                sequentials.append(
                    nn.Sequential(Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
                    Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
                )
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 3, 2, 1)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 3, 2, 1)
                face_encoder_channels.append(channels * 2)
            elif i == n_layers - 1:
                estimate = evaluate_new_size_after_conv(evaluate_new_size_after_conv(last_face_x_size, 3, 1, 0), 1, 1, 0)
                
                if estimate == 1:
                    sequentials.append(
                        nn.Sequential(Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
                        Conv2d(channels, channels, kernel_size=1, stride=1, padding=0))
                    )
                    last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 3, 1, 0)
                    last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 3, 1, 0)
                    last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 1, 1, 0)
                    last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 1, 1, 0)
                else:
                    sequentials.append(
                        nn.Sequential(Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
                        Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
                        Conv2d(channels, channels, kernel_size=1, stride=1, padding=0))
                    )
                    last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 3, 2, 1)
                    last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 3, 2, 1)
                    last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 3, 1, 0)
                    last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 3, 1, 0)
                    last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 1, 1, 0)
                    last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 1, 1, 0)
                    
                face_encoder_channels.append(channels)
            else:
                sequentials.append(
                    nn.Sequential(Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
                    Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, residual=True))
                )
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 3, 2, 1)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 3, 2, 1)
                face_encoder_channels.append(channels * 2)
            channels *= 2
            
            print("[wav2lip] face_encoder_blocks x, y", last_face_x_size, last_face_y_size)
        assert last_face_x_size == 1
        assert last_face_y_size == 1

        self.face_encoder_blocks = nn.ModuleList(sequentials)

        face_final_channels = channels // 2
        audio_layers = int(np.log(face_final_channels) / np.log(2) - 4)

        self.audio_encoder, audio_shapes = create_audio_encoder(audio_layers, hp.batch_size, for_wav2lip=True)
        print("[wav2lip] review audio_encoder shapes")
        print("[wav2lip] face_final_channels after encode", face_final_channels)
        print("[wav2lip] face encoder block channels", face_encoder_channels)
        print("[wav2lip] audio_shapes")
        print(*audio_shapes, sep='\n')


        n_layers = evaluate_conv_layers(hp.img_size) + 1
        sequentials = []
        input_channels = face_final_channels

        last_face_x_size = 1
        last_face_y_size = 1

        rev_face_encoder_blocks_channels = list(reversed(face_encoder_channels))
        FIXED_OUTPUT_CHANNELS = [64, 128, 256, 384, 512, 1024, 2048]
        required_output_channels = FIXED_OUTPUT_CHANNELS[:(n_layers - 2)]
        required_output_channels += [required_output_channels[-1]] * 2
        required_output_channels = list(reversed(required_output_channels))
        print("[wav2lip] required output channels", required_output_channels)

        for i in range(n_layers):
            if i == 0:
                sequentials.append(
                    nn.Sequential(Conv2d(input_channels, required_output_channels[i], kernel_size=1, stride=1, padding=0),)
                )
            elif i == 1:
                sequentials.append(
                    nn.Sequential(Conv2dTranspose(input_channels, required_output_channels[i], kernel_size=3, stride=1, padding=0), # 3,3
                    Conv2d(required_output_channels[i], required_output_channels[i], kernel_size=3, stride=1, padding=1, residual=True),)
                )
            else:
                sequentials.append(
                    nn.Sequential(Conv2dTranspose(input_channels, required_output_channels[i], kernel_size=3, stride=2, padding=1, output_padding=1),
                    Conv2d(required_output_channels[i], required_output_channels[i], kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(required_output_channels[i], required_output_channels[i], kernel_size=3, stride=1, padding=1, residual=True),)
                )
            input_channels = required_output_channels[i] + rev_face_encoder_blocks_channels[i]

        self.face_decoder_blocks = nn.ModuleList(sequentials)

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()) 

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            
            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x
            
        return outputs # (BxT, 3, img_size, img_size)

class Wav2Lip_disc_qual(nn.Module):
    def __init__(self):
        super(Wav2Lip_disc_qual, self).__init__()

        n_layers = evaluate_conv_layers(hp.img_size) + 1
        sequentials = []
        FIXED_OUTPUT_CHANNELS = [32, 64, 128, 256, 512, 1024, 2048]
        required_output_channels = FIXED_OUTPUT_CHANNELS[:(n_layers - 2)]
        required_output_channels += [required_output_channels[-1]] * 2

        last_face_x_size = hp.img_size // 2
        last_face_y_size = hp.img_size

        input_channels = 3
        for i in range(n_layers):
            if i == 0:
                sequentials.append(
                    nn.Sequential(nonorm_Conv2d(input_channels, required_output_channels[i], kernel_size=7, stride=1, padding=3))
                )
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 7, 1, 3)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 7, 1, 3)
            elif i == 1:
                sequentials.append(
                nn.Sequential(nonorm_Conv2d(input_channels, required_output_channels[i], kernel_size=5, stride=(1, 2), padding=2),
                nonorm_Conv2d(required_output_channels[i], required_output_channels[i], kernel_size=5, stride=1, padding=2)),)

                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 5, 1, 2)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 5, 2, 2)
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 5, 1, 2)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 5, 1, 2)
            elif i < n_layers - 3:
                sequentials.append(
                nn.Sequential(nonorm_Conv2d(input_channels, required_output_channels[i], kernel_size=5, stride=2, padding=2),
                nonorm_Conv2d(required_output_channels[i], required_output_channels[i], kernel_size=5, stride=1, padding=2)))

                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 5, 2, 2)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 5, 2, 2)
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 5, 1, 2)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 5, 1, 2)
            elif i == n_layers - 1:
                sequentials.append(
                    nn.Sequential(nonorm_Conv2d(input_channels, required_output_channels[i], kernel_size=3, stride=1, padding=0),
                    nonorm_Conv2d(required_output_channels[i], required_output_channels[i], kernel_size=1, stride=1, padding=0)),
                )
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 3, 1, 0)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 3, 1, 0)
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 1, 1, 0)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 1, 1, 0)
            elif i == n_layers - 2:
                sequentials.append(
                    nn.Sequential(nonorm_Conv2d(input_channels, required_output_channels[i], kernel_size=3, stride=2, padding=1),
                    nonorm_Conv2d(required_output_channels[i], required_output_channels[i], kernel_size=3, stride=1, padding=1),),
                )
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 3, 2, 1)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 3, 2, 1)
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 3, 1, 1)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 3 ,1 ,1)
                
            else:
                sequentials.append(
                    nn.Sequential(nonorm_Conv2d(input_channels, required_output_channels[i], kernel_size=3, stride=2, padding=1),
                    nonorm_Conv2d(required_output_channels[i], required_output_channels[i], kernel_size=3, stride=1, padding=1)),
                )
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 3, 2, 1)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 3, 2, 1)
                last_face_x_size = evaluate_new_size_after_conv(last_face_x_size, 3, 1, 1)
                last_face_y_size = evaluate_new_size_after_conv(last_face_y_size, 3, 1, 1)
            input_channels = required_output_channels[i]

        self.face_encoder_blocks = nn.ModuleList(sequentials)
        assert last_face_x_size == 1, last_face_x_size
        assert last_face_y_size == 1, last_face_y_size
        final_channels = input_channels

        self.binary_pred = nn.Sequential(nn.Conv2d(final_channels, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2)//2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1), 
                                        torch.ones((len(false_feats), 1)).cuda())

        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)
