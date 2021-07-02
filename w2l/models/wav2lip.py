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

        self.output_block = nn.Sequential(Conv2d(108, 32, kernel_size=3, stride=1, padding=1),  # (+16)
                                          nn.Conv2d(32, 3, kernel_size=1,
                                                    stride=1, padding=0),
                                          nn.Sigmoid())
        self.landmarks_decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(512, len(hp.landmarks_points) * 2),
        )

    def dump_face(self, face_sequences, dump_dir):
        from uuid import uuid4
        import os
        import cv2
        import numpy as np

        print("wav2lip dump faces to dir:", dump_dir)
        B = face_sequences.size(0)
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            dump = (face_sequences.reshape((B, hp.syncnet_T, 6, hp.img_size,
                    hp.img_size)) * 255.).detach().cpu().numpy().astype(np.uint8)
        else:
            dump = (face_sequences.reshape((B, 6, hp.img_size, hp.img_size))
                    * 255.).detach().cpu().numpy().astype(np.uint8)
        hex = uuid4().hex
        for b in range(B):
            if input_dim_size > 4:
                for t in range(6):
                    img = dump[b, t, :3]  # (3, H, W)
                    img = img.transpose((1, 2, 0))
                    filename = os.path.join(
                        dump_dir, f'wav2lip_{hex}_real_{b}-{t}.jpg')
                    cv2.imwrite(filename, img)

                    img = dump[b, t, 3:]  # (3, H, W)
                    img = img.transpose((1, 2, 0))
                    filename = os.path.join(
                        dump_dir, f'wav2lip_{hex}_fake_{b}-{t}.jpg')
                    cv2.imwrite(filename, img)
            else:
                img = dump[b, :3]  # (3, H, W)
                img = img.transpose((1, 2, 0))
                filename = os.path.join(
                    dump_dir, f'wav2lip_{hex}_real_{b}.jpg')
                cv2.imwrite(filename, img)

                img = dump[b, 3:]  # (3, H, W)
                img = img.transpose((1, 2, 0))
                filename = os.path.join(
                    dump_dir, f'wav2lip_{hex}_fake_{b}.jpg')
                cv2.imwrite(filename, img)

    def forward(self, audio_sequences, face_sequences):
        # self.dump_face(face_sequences, "/hdd/checkpoints/w2l/temp")

        # face_sequences: (B, T, 6, H, W)
        # audio_sequences: (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = audio_sequences.reshape(
                (B * hp.syncnet_T, 1, hp.num_mels, hp.syncnet_mel_step_size))
            face_sequences = face_sequences.reshape(
                (B * hp.syncnet_T, 6, hp.img_size, hp.img_size))

        # face_sequences: (B x T, 6, H, W)
        audio_embedding = self.audio_encoder(
            audio_sequences)  # B x T, 512, 1, 1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        # (B x T, 512, 1, 1)
        face_embedding = x

        # (B x T, 512, 1, 1)
        x = audio_embedding
        if input_dim_size > 4:
            # (B x T, 1024, 1, 1)
            embedding = torch.cat([face_embedding, audio_embedding], dim=1).reshape(
                (B * hp.syncnet_T, 1024))

            # (B, T, 14, 2)
            landmarks = self.landmarks_decoder(
                embedding).reshape((B, hp.syncnet_T, 14, 2))
        else:
            landmarks = None

        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                feat = feats.pop()
                if len(feats) == 0:
                    feat = feat[:, :, hp.half_img_size:]
                x = torch.cat((x, feat), dim=1)
            except Exception as e:
                print("x", x.size(), "feat", feat.size())
                raise e

        x = self.output_block(x)

        if input_dim_size > 4:
            x = x.reshape((B, hp.syncnet_T, 3, hp.half_img_size, hp.img_size))

        # (B, T, 3, half_img_size, img_size), (B, T, 14, 2)
        return x, landmarks


class Wav2Lip_disc_qual(nn.Module):
    def __init__(self):
        super(Wav2Lip_disc_qual, self).__init__()

        # input (x, y) or (x, y_hat)
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=4, stride=2, padding=1),  # 96,192
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 96,96
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),    # 48,48
            nn.GELU(),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

        # self.face_encoder = nn.Sequential(
        #     nonorm_Conv2d(6, 32, kernel_size=7, stride=1, padding=3),  # 96,192
        #     nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),  # 96,96
        #     nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
        #     nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),    # 48,48
        #     nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        #     nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),   # 24,24
        #     nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
        #     nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),       # 12,12
        #     nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),     # 6,6
        #     nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 4, 4
        #     nonorm_Conv2d(512, 1, kernel_size=1, stride=1, padding=0),
        #     nn.Sigmoid(),
        # )

    def dump_faces(self, face_sequences, dump_dir):
        from uuid import uuid4
        import os
        import cv2
        import numpy as np

        print("discriminator dump faces to dir", dump_dir)
        B = face_sequences.size(0)
        dump = (face_sequences.reshape((B, hp.syncnet_T, 3, hp.half_img_size,
                hp.img_size)) * 255.).detach().cpu().numpy().astype(np.uint8)
        hex = uuid4().hex
        for b in range(B):
            for t in range(6):
                img = dump[b, t, :]  # (3, H, W)
                img = img.transpose((1, 2, 0))
                filename = os.path.join(dump_dir, f'disc_{hex}_{b}-{t}.jpg')
                cv2.imwrite(filename, img)

    def to_2d(self, face_sequences):
        # face_sequences: (B, T, 3, H, W)

        # self.dump_faces(face_sequences, '/hdd/checkpoints/w2l/temp')

        B = face_sequences.size(0)

        # (B x T, 3, H, W)
        face_sequences = face_sequences.reshape(
            (B * hp.syncnet_T, 9, hp.half_img_size, hp.img_size))
        return face_sequences

    def forward_(self, half_face_sequences, half_input):
        half_face_sequences = torch.cat([half_input, half_face_sequences], dim=2)
        x = self.to_2d(half_face_sequences)
        return self.patch_encoder(x)

    def perceptual_forward(self, false_half_face_sequences, input):
        false_activation = self.forward_(false_half_face_sequences, input)
        # print("false_activation", false_activation.shape)

        B = false_activation.size(0)

        false_pred_loss = F.binary_cross_entropy(
            false_activation, torch.ones((B, 1, 10, 22), device='cuda'), reduction='none').mean([2, 3]).reshape((B,))

        return false_pred_loss

    def forward(self, half_face_sequences, input):
        return self.forward_(half_face_sequences, input)


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
        B = face_sequences.size(0)

        # (B x T, 3, H, W)
        face_sequences = face_sequences.reshape(
            (B * hp.syncnet_T, 3, hp.half_img_size, hp.img_size))
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
