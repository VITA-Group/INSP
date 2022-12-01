import cv2
import glob2
import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import matplotlib.pyplot as plt
import time

from functools import partial

import dataio, modules

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

li = ['./logs/audio_noisy_10/checkpoints/model_current.pth']
print(len(li))

# data
sz = 64
audio_dataset = dataio.AudioFile(filename='data/gt_bach.wav')
coord_dataset = dataio.ImplicitAudioWrapper_ray(audio_dataset)

dataloader = DataLoader(coord_dataset, shuffle=False, batch_size=1024, pin_memory=False, num_workers=4)

device = torch.device('cuda:0')

model = modules.SingleBVPNet(type='sine', mode='mlp', in_features=1).cuda()

# @torch.no_grad()
def extract_image(model, dataloader, device):
    output = []
    for step, (model_input, gt) in enumerate(tqdm.tqdm(dataloader)):
        model_input = {key: value.to(device) for key, value in model_input.items() if key != 'ckpt'}
        model_output = model(model_input)
        output.append(model_output['model_out'].detach().cpu().numpy())
    output = np.concatenate(output, 1)
    return output

import diff_operators


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def new_grad_audio(y, x, num=3, sz=256):
    li = [y.squeeze(-1)]
    # print(y.max())
    for i in range(num):
        cur = li[i]
        ww = torch.autograd.grad(cur / 256, x, torch.ones_like(cur), create_graph=True)[0]
        li.append(ww[..., 0])
    return torch.stack([*li], dim=0).unsqueeze(-1)


def grad_model(model, model_input):
    model_output = model(model_input)
    y, x = model_output['model_out'], model_output['model_in']
    new = new_grad_audio(y, x, num=7)
    return {'model_out': new}

def process(fname):
    model.load_state_dict(torch.load(fname))
    g_model = lambda model_input: grad_model(model, model_input)
    g_output = extract_image(g_model, dataloader, device).squeeze(-1)
    print(g_output.shape)
    new = os.path.join('./grad/audio_grad', os.path.basename(fname.split('/')[-3]) + '.npy')
    print(new)
    np.save(new, g_output)

num = len(li)
li = sorted(li)
for cur in tqdm.tqdm(li):
    process(cur)

