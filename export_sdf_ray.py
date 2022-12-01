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
# import skimage
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

li = ['./data/dragon_50000.ckpt']
print(len(li))

# data
sz = 256
coord_dataset = dataio.SDFWrapper_ray()
image_resolution = (sz, sz)
dataloader = DataLoader(coord_dataset, shuffle=False, batch_size=1024, pin_memory=False, num_workers=8)
# len is num of image

device = torch.device('cuda:0')

import inrnet

model = inrnet.INRNet(out_features=1, in_features=3, num_hidden_layers=2, hidden_features=256, pos_emb='Id', nonlinearity='sine').cuda()

def extract_image(model, dataloader, image_resolution, device):
    output = []
    for step, (model_input, gt) in enumerate(tqdm.tqdm(dataloader)):
        # for k, v in model_input.items():
        #     print(k, type(v))
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

def new_grad(y, x, sz=256):
    li = [y.squeeze(-1)]
    for i in range(7 * 3):
        cur = li[i]
        ww = torch.autograd.grad(cur / sz, x, torch.ones_like(cur), create_graph=True)[0]
        li.append(ww[..., 0])
        li.append(ww[..., 1])
        li.append(ww[..., 2])
    return torch.stack(li, dim=0).unsqueeze(-1)

def grad_model(model, model_input):
    model_output = model(model_input)
    y, x = model_output['model_out'], model_output['model_in']
    new = new_grad(y, x)
    return {'model_out': new}

def process(fname):
    model.load_state_dict(torch.load(fname, map_location=torch.device('cpu'))['model'])
    g_model = lambda model_input: grad_model(model, model_input)
    g_output = extract_image(g_model, dataloader, image_resolution, device).squeeze(-1)
    print(g_output.shape)
    # print(fname.split('/')[-3])
    new = os.path.join('./grad/sdf_grad/dragon_siren.npy')
    print(new)
    np.save(new, g_output)

num = len(li)
li = sorted(li)
for cur in tqdm.tqdm(li):
    process(cur)

