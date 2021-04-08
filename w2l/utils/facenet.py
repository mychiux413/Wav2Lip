import os
from w2l.hparams import hparams
import torch
import torchvision
from w2l.models.mobilefacenet import MobileFaceNet
from w2l.utils.env import device


def load_facenet_model():
    if torch.cuda.is_available():
        def map_location(storage, loc): return storage.cuda()
    else:
        map_location = 'cpu'
    assert os.path.exists(hparams.mobilefacenet_model_path)
    model = MobileFaceNet([112, 112], 136)
    checkpoint = torch.load(
        hparams.mobilefacenet_model_path,
        map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model = model.eval()
    return model


resize_for_facenet = torchvision.transforms.Resize([112, 112])


def mask_mouth(model, window, reverse=False):
    new_window = window.detach().clone()
    window = resize_for_facenet(window).transpose(0, 1)
    landmarks = model(window.to(device)).cpu()[0]
    landmarks = landmarks.reshape(hparams.syncnet_T, -1, 2)
    mouth_landmarks = landmarks[:, 49:, :]
    for i, landmark in enumerate(mouth_landmarks):
        mouth_x1 = int(min(landmark[:, 0]) * hparams.img_size)
        mouth_x2 = int(max(landmark[:, 0]) * hparams.img_size)
        mouth_y1 = int(min(landmark[:, 1]) * hparams.img_size)
        mouth_y2 = int(max(landmark[:, 1]) * hparams.img_size)
        mouth_width = mouth_x2 - mouth_x1
        mouth_height = mouth_y2 - mouth_y1
        mouth_x1 = max(0, int(mouth_x1 - mouth_width *
                       hparams.expand_mouth_width_ratio))
        mouth_x2 = min(hparams.img_size, int(
            mouth_x2 + mouth_width * hparams.expand_mouth_width_ratio))
        mouth_y1 = max(0, int(mouth_y1 - mouth_height *
                       hparams.expand_mouth_height_ratio))
        mouth_y2 = min(hparams.img_size, int(
            mouth_y2 + mouth_height * hparams.expand_mouth_height_ratio))
        if reverse:
            mask = torch.zeros((1,) + new_window.shape[2:])
            mask[:, mouth_y1:mouth_y2, mouth_x1:mouth_x2] = 1.
            new_window[:, i] *= mask
        else:
            new_window[:, i, mouth_y1:mouth_y2, mouth_x1:mouth_x2] = 0.
    return new_window
