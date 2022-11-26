import cv2
import glob2
import tqdm

import torch
# torch.set_default_dtype(torch.float16)
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

import configargparse
p = configargparse.ArgumentParser()
# p.add_argument('--c', required=True, type=int)
p.add_argument('--save_dir', required=True, type=str)
# /home/dejia/repo/siren/data/div2k_color_hole_grad
p.add_argument('--load', required=True, type=str)
p.add_argument('--offset', default=0, type=int)
p.add_argument('--single', action='store_true')
# div2k_*.png_color_hole
args = p.parse_args()
os.makedirs(args.save_dir, exist_ok=True)

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

li = glob2.glob(f'./logs/{args.load}/checkpoints/model_final.pth')
print(len(li))

# data
sz = 256
img_dataset = dataio.NoisyCamera_multimlp(img_path='div2k', img_num=1)
coord_dataset = dataio.Implicit2DWrapper_multimlp_ray(img_dataset, sidelength=sz, compute_diff='blur_x', sigma=5)
image_resolution = (sz, sz)
dataloader = DataLoader(coord_dataset, shuffle=False, batch_size=4096, pin_memory=False, num_workers=4)
# len is num of image

device = torch.device('cuda:0')
model = modules.SingleBVPNet(type='sine', mode='mlp', sidelength=image_resolution, hidden_features=256, num_hidden_layers=3, out_features=3).to(device)

def extract_image(model, dataloader, image_resolution, device):
    output = []
    for step, (model_input, gt) in enumerate((dataloader)):
        model_input = {key: value.to(device) for key, value in model_input.items() if key != 'ckpt'}
        model_output = model(model_input)
        output.append(model_output['model_out'].detach().cpu().numpy())
    output = np.concatenate(output, 0)
    return output

import diff_operators


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def new_grad_lastdim(y, x, sz=256, num=31):
    li = [y[..., 0], y[..., 1], y[..., 2]]
    for i in range(num):
        cur = li[i]
        ww = torch.autograd.grad(cur / sz, x, torch.ones_like(cur), create_graph=True)[0]
        li.append(ww[..., 0])
        li.append(ww[..., 1])
    return torch.stack(li, dim=-1)

def grad_model(model, model_input):
    model_output = model(model_input)
    y, x = model_output['model_out'], model_output['model_in']
    new = new_grad_lastdim(y, x, 63 * 3)
    return {'model_out': new}

def process(fname):
    model.load_state_dict(torch.load(fname))
    g_model = lambda model_input: grad_model(model, model_input)
    g_output = extract_image(g_model, dataloader, image_resolution, device)#.squeeze(-1)
    print(g_output.shape)
    new = os.path.join(args.save_dir, os.path.basename(fname.split('/')[-3]).replace('.png', f'.npy'))
    np.save(new, g_output)

num = len(li)
if args.single:
    li = sorted(li)[args.offset:args.offset + 1]
else:
    li = sorted(li)[args.offset:]
print(li)
for cur in tqdm.tqdm(li):
    process(cur)

