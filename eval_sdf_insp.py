
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

name = 'dragon'

model = INSP(64, 1).cuda()
model.load_state_dict(torch.load(f'./logs/smooth_{name}/checkpoints/model_current.pth'))
model.eval()

import dataio

import numpy as np
grad = np.load(f'./grad/sdf_grad/{name}_siren.npy')
for i in range(grad.shape[0]):
    # print(grad[i].max())
    while (grad[i].max() > 10):
        grad[i] /= 256

grad = torch.from_numpy(grad).cuda()
print(grad.shape)
import time
st = time.time()
with torch.no_grad():
    # out = model({'grad': grad})
    out = model({'grad': grad.permute(1, 0)})
    out = out['new_img'].detach().cpu().reshape(256, 256, 256)#.numpy()
    print(out.max())

ed = time.time()
print(ed - st)

import sdf_meshing
sdf_meshing.convert_sdf_samples_to_ply(out, [-1, -1, -1], 2 / 256, f'output/sdf/res_insp_{name}_10.ply')
np.save(f'output/sdf/sdf_insp_{name}.npy', out)
