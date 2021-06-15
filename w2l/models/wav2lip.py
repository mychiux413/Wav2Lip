import torch
from torch import nn
from torch.nn import functional as F
from w2l.hparams import hparams as hp

from w2l.models.conv import Conv2dTranspose, Conv2d, nonorm_Conv2d, evaluate_conv_layers, evaluate_new_size_after_conv, create_audio_encoder
from w2l.models.mobilefacenet import BatchNorm1d, Flatten, Linear
import torchvision


class GDC(nn.Module):
    def __init__(self, channels, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_flatten = Flatten()
        self.linear = Linear(channels, embedding_size, bias=False)
        # self.bn = BatchNorm1d(embedding_size, affine=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        # BxT, channels, 1, 1
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        # BxT, 76
        x = self.bn(x)
        return x


class Wav2Lip(nn.Module):
    def __init__(self):
        super(Wav2Lip, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7,
                          stride=1, padding=3)),  # 192,192

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 96,96
                          Conv2d(32, 32, kernel_size=3, stride=1,
                                 padding=1, residual=True),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # 48,48
                          Conv2d(64, 64, kernel_size=3, stride=1,
                                 padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1,
                                 padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 24,24
                          Conv2d(128, 128, kernel_size=3, stride=1,
                                 padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),       # 12,12
                          Conv2d(256, 256, kernel_size=3, stride=1,
                                 padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 384, kernel_size=3, stride=2, padding=1),       # 6,6
                          Conv2d(384, 384, kernel_size=3, stride=1,
                                 padding=1, residual=True),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(384, 512, kernel_size=3, stride=2, padding=1),     # 3,3
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
                          Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1,
                   padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1,
                   padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1,
                   padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(
                Conv2d(512, 512, kernel_size=1, stride=1, padding=0),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),  # 3,3  (+512)
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(512, 512, kernel_size=3, stride=1,
                                 padding=1, residual=True),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),  # 6, 6    (+512)

            nn.Sequential(Conv2dTranspose(896, 448, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(448, 448, kernel_size=3, stride=1,
                                 padding=1, residual=True),
                          Conv2d(448, 448, kernel_size=3, stride=1, padding=1, residual=True),),  # 12, 12  (+384)

            nn.Sequential(Conv2dTranspose(704, 352, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(352, 352, kernel_size=3, stride=1,
                                 padding=1, residual=True),
                          Conv2d(352, 352, kernel_size=3, stride=1, padding=1, residual=True),),  # 24, 24  (+256)

            nn.Sequential(Conv2dTranspose(480, 240, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(240, 240, kernel_size=3, stride=1,
                                 padding=1, residual=True),
                          Conv2d(240, 240, kernel_size=3, stride=1, padding=1, residual=True),),  # 48, 48  (+128)

            nn.Sequential(Conv2dTranspose(304, 152, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(152, 152, kernel_size=3, stride=1,
                                 padding=1, residual=True),
                          Conv2d(152, 152, kernel_size=3, stride=1, padding=1, residual=True),),  # 96,96  (+64)


            nn.Sequential(Conv2dTranspose(184, 92, kernel_size=3, stride=(1, 2), padding=1, output_padding=(0, 1)),
                          Conv2d(92, 92, kernel_size=3, stride=1,
                                 padding=1, residual=True),
                          Conv2d(92, 92, kernel_size=3, stride=1, padding=1, residual=True),), ])   # 192,192  (+32)

        self.output_block = nn.Sequential(Conv2d(108, 32, kernel_size=3, stride=1, padding=1), # (+16)
                                          nn.Conv2d(32, 3, kernel_size=1,
                                                    stride=1, padding=0),
                                          nn.Sigmoid())

    def forward(self, audio_sequences, face_sequences):
        # face_sequences: (B, 6, T, H, W)
        # audio_sequences: (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat(
                [audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat(
                [face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        # face_sequences: (B x T, 6, H, W)
        audio_embedding = self.audio_encoder(
            audio_sequences)  # B x T, 512, 1, 1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                feat = feats.pop()
                if len(feats) == 0:
                    feat = feat[:, :, hp.img_size // 2:]
                x = torch.cat((x, feat), dim=1)
            except Exception as e:
                print("x", x.size(), "feat", feat.size())
                raise e

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)  # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)

        else:
            outputs = x

        return outputs  # (BxT, 3, img_size, img_size)


class Wav2Lip_disc_qual(nn.Module):
    def __init__(self):
        super(Wav2Lip_disc_qual, self).__init__()

        n_layers = evaluate_conv_layers(hp.img_size) + 1
        sequentials = []
        FIXED_OUTPUT_CHANNELS = [32, 64, 128, 256, 368, 512, 512]
        required_output_channels = FIXED_OUTPUT_CHANNELS[:(n_layers - 2)]
        required_output_channels += [required_output_channels[-1]] * 2

        last_face_x_size = hp.img_size // 2
        last_face_y_size = hp.img_size

        input_channels = 3
        for i in range(n_layers):
            if i == 0:
                sequentials.append(
                    nn.Sequential(nonorm_Conv2d(
                        input_channels, required_output_channels[i], kernel_size=7, stride=1, padding=3))
                )
                last_face_x_size = evaluate_new_size_after_conv(
                    last_face_x_size, 7, 1, 3)
                last_face_y_size = evaluate_new_size_after_conv(
                    last_face_y_size, 7, 1, 3)
            elif i == 1:
                sequentials.append(
                    nn.Sequential(nonorm_Conv2d(input_channels, required_output_channels[i], kernel_size=5, stride=(1, 2), padding=2),
                                  nonorm_Conv2d(required_output_channels[i], required_output_channels[i], kernel_size=5, stride=1, padding=2)),)

                last_face_x_size = evaluate_new_size_after_conv(
                    last_face_x_size, 5, 1, 2)
                last_face_y_size = evaluate_new_size_after_conv(
                    last_face_y_size, 5, 2, 2)
                last_face_x_size = evaluate_new_size_after_conv(
                    last_face_x_size, 5, 1, 2)
                last_face_y_size = evaluate_new_size_after_conv(
                    last_face_y_size, 5, 1, 2)
            elif i < n_layers - 3:
                sequentials.append(
                    nn.Sequential(nonorm_Conv2d(input_channels, required_output_channels[i], kernel_size=5, stride=2, padding=2),
                                  nonorm_Conv2d(required_output_channels[i], required_output_channels[i], kernel_size=5, stride=1, padding=2)))

                last_face_x_size = evaluate_new_size_after_conv(
                    last_face_x_size, 5, 2, 2)
                last_face_y_size = evaluate_new_size_after_conv(
                    last_face_y_size, 5, 2, 2)
                last_face_x_size = evaluate_new_size_after_conv(
                    last_face_x_size, 5, 1, 2)
                last_face_y_size = evaluate_new_size_after_conv(
                    last_face_y_size, 5, 1, 2)
            elif i == n_layers - 1:
                sequentials.append(
                    nn.Sequential(nonorm_Conv2d(input_channels, required_output_channels[i], kernel_size=3, stride=1, padding=0),
                                  nonorm_Conv2d(required_output_channels[i], required_output_channels[i], kernel_size=1, stride=1, padding=0)),
                )
                last_face_x_size = evaluate_new_size_after_conv(
                    last_face_x_size, 3, 1, 0)
                last_face_y_size = evaluate_new_size_after_conv(
                    last_face_y_size, 3, 1, 0)
                last_face_x_size = evaluate_new_size_after_conv(
                    last_face_x_size, 1, 1, 0)
                last_face_y_size = evaluate_new_size_after_conv(
                    last_face_y_size, 1, 1, 0)
            elif i == n_layers - 2:
                sequentials.append(
                    nn.Sequential(nonorm_Conv2d(input_channels, required_output_channels[i], kernel_size=3, stride=2, padding=1),
                                  nonorm_Conv2d(required_output_channels[i], required_output_channels[i], kernel_size=3, stride=1, padding=1),),
                )
                last_face_x_size = evaluate_new_size_after_conv(
                    last_face_x_size, 3, 2, 1)
                last_face_y_size = evaluate_new_size_after_conv(
                    last_face_y_size, 3, 2, 1)
                last_face_x_size = evaluate_new_size_after_conv(
                    last_face_x_size, 3, 1, 1)
                last_face_y_size = evaluate_new_size_after_conv(
                    last_face_y_size, 3, 1, 1)

            else:
                sequentials.append(
                    nn.Sequential(nonorm_Conv2d(input_channels, required_output_channels[i], kernel_size=3, stride=2, padding=1),
                                  nonorm_Conv2d(required_output_channels[i], required_output_channels[i], kernel_size=3, stride=1, padding=1)),
                )
                last_face_x_size = evaluate_new_size_after_conv(
                    last_face_x_size, 3, 2, 1)
                last_face_y_size = evaluate_new_size_after_conv(
                    last_face_y_size, 3, 2, 1)
                last_face_x_size = evaluate_new_size_after_conv(
                    last_face_x_size, 3, 1, 1)
                last_face_y_size = evaluate_new_size_after_conv(
                    last_face_y_size, 3, 1, 1)
            input_channels = required_output_channels[i]

        self.face_encoder_blocks = nn.ModuleList(sequentials)
        assert last_face_x_size == 1, last_face_x_size
        assert last_face_y_size == 1, last_face_y_size
        final_channels = input_channels

        self.binary_pred = nn.Sequential(
            nn.Conv2d(final_channels, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    # def get_lower_half(self, face_sequences):
    #     return face_sequences[:, :, face_sequences.size(2)//2:]

    def to_2d(self, face_sequences):
        # B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i]
                                   for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def forward_(self, half_face_sequences):
        x = self.to_2d(half_face_sequences)
        for f in self.face_encoder_blocks:
            x = f(x)
        return x

    def perceptual_forward(self, false_half_face_sequences):
        false_feats = self.forward_(false_half_face_sequences)

        false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1),
                                                 torch.ones((len(false_feats), 1)).cuda())

        return false_pred_loss

    def forward(self, half_face_sequences):
        x = self.forward_(half_face_sequences)

        return self.binary_pred(x).view(len(x), -1)


inception_resize = torchvision.transforms.Resize((299, 299))


class InceptionV3_disc(torchvision.models.Inception3):

    def __init__(self, pretrained=False):
        super().__init__(num_classes=1, aux_logits=False, init_weights=True)

        if pretrained:
            pretrained_model = torchvision.models.inception_v3(
                pretrained=True, progress=False, aux_logits=False)
            pretrained_dict = pretrained_model.state_dict()
            pretrained_dict.pop('fc.weight')
            pretrained_dict.pop('fc.bias')
            self.load_state_dict(pretrained_dict, strict=False)

    def to_2d(self, face_sequences):
        # B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i]
                                   for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def forward_(self, half_face_sequences):
        x = self.to_2d(half_face_sequences)
        return torch.sigmoid(super().forward(inception_resize(x)))

    def perceptual_forward(self, false_half_face_sequences):

        false_feats = self.forward_(false_half_face_sequences)

        false_pred_loss = F.binary_cross_entropy(false_feats,
                                                 torch.ones((len(false_feats), 1)).cuda())

        return false_pred_loss

    def forward(self, half_face_sequences):
        return self.forward_(half_face_sequences)
