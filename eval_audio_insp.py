import configargparse, cv2
from functools import partial
from dataio import get_mgrid
import modules, diff_operators
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from collections import OrderedDict
import os

class INSP(nn.Module):
    def __init__(self, in_c=3*23, out_c=3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_c, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, out_c)
    def forward(self, data):
        x = data['grad']
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.fc2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.fc3(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.fc5(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.fc4(x)
        return {'new_img': x}

model = INSP(8, 1).cuda()
model.load_state_dict(torch.load('./logs/audio_denoise/checkpoints/model_current.pth'))
model.eval()

import dataio

import numpy as np
grad = np.load('./grad/audio_grad/audio_noisy_10.npy')
for i in range(grad.shape[0]):
    while (grad[i].max() > 10):
        grad[i] /= 256

grad = torch.from_numpy(grad).cuda()
with torch.no_grad():
    out = model({'grad': grad.permute(1, 0)})
    out = out['new_img'].detach().cpu()
    print(out.max())

import scipy.io.wavfile as wavfile
wavfile.write('audio_insp.wav', 44100, out.numpy())
