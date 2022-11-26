
import configargparse, cv2
from functools import partial
from dataio import get_mgrid
import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from collections import OrderedDict
import os

p = configargparse.ArgumentParser()
p.add_argument('--save_path', required=True, type=str)
p.add_argument('--target', required=True, type=str)
p.add_argument('--ckpt_path', required=True, type=str)
args = p.parse_args()

os.makedirs(args.save_path, exist_ok=True)

class INSP(nn.Module):
    def __init__(self, sz=127 * 3, in_c=3, out_c=3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(65, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, out_c)
    def forward(self, data):
        x = data['grad']
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0, inplace=True)
        x = self.fc2(x)
        x = F.leaky_relu(x, negative_slope=0, inplace=True)
        x = self.fc3(x)
        x = F.leaky_relu(x, negative_slope=0, inplace=True)
        x = self.fc5(x)
        x = F.leaky_relu(x, negative_slope=0, inplace=True)
        x = self.fc4(x)
        return {'new_img': x}

model = INSP().cuda()
model.load_state_dict(torch.load(args.ckpt_path))
model.eval()

if args.target == 'blur':
    base = './grad/test_color_ori/'
    suff = '_ori'
elif args.target == 'deblur':
    base = './grad/test_color_blur/'
    suff = '_blur'
elif args.target == 'denoise':
    base = './grad/test_color_noise/'
    suff = '_noise'
elif args.target == 'inpainting':
    base = './grad/test_color_hole/'
    suff = '_hole'
elif args.target == 'inpainting_text':
    base = './grad/test_color_inpainting_text/'
    suff = '_inpainting_text'
import dataio

import glob2, numpy as np
import tqdm
from IPython import embed
grad_li = glob2.glob(base + f'/*{suff}.npy')
grad_li = sorted(grad_li)
for idx, cur in enumerate(grad_li):
  grad = np.load(cur)
  grad = grad.transpose(1, 0)
#   print(grad.shape, '??')
  for i in range(grad.shape[0]):
    while (grad[i].max() > 10):
        grad[i] /= 256
  grad = torch.from_numpy(grad).cuda()
#   print(grad.shape)
  with torch.no_grad():
    out = model({'grad': grad.permute(1, 0)})
    out = out['new_img'].detach().cpu().view(256, 256, 3).numpy() * 255
    print(out.max())
  ori = torch.stack([grad[:3]]).view(3, 256, 256).permute(1, 2, 0)
  ori = ori.detach().cpu().numpy() * 255 # * 255
  ori = np.clip(ori, 0, 255)
  out = np.clip(out, 0, 255)
  print(ori.max(), ori.shape, out.shape)
  fname = os.path.join(args.save_path, f"{os.path.basename(cur).replace('.npy', '')}_ori.png")
  Image.fromarray(ori.astype(np.uint8)).convert('RGB').save(fname)
  Image.fromarray(out.astype(np.uint8)).convert('RGB').save(fname.replace('_ori.png', '_out.png'))
  print(fname)
  print(out.shape)

