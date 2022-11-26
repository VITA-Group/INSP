import csv
import glob
import math
import os

import matplotlib.colors as colors
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.ndimage
import scipy.special
import skimage
import skimage.filters
import skvideo.io
import torch
from PIL import Image, ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from kornia.filters import sobel

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

def get_3d_mgrid(shape):
    pixel_coords = np.stack(np.mgrid[:shape[0], :shape[1], :shape[2]], axis=-1).astype(np.float32)

    # normalize pixel coords onto [-1, 1]
    pixel_coords[..., 0] = pixel_coords[..., 0] / max(shape[0] - 1, 1)
    pixel_coords[..., 1] = pixel_coords[..., 1] / max(shape[1] - 1, 1)
    pixel_coords[..., 2] = pixel_coords[..., 2] / max(shape[2] - 1, 1)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    # flatten 
    pixel_coords = torch.tensor(pixel_coords).view(-1, 3)

    return pixel_coords

def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


def grads2img(gradients):
    mG = gradients.detach().squeeze(0).permute(-2, -1, -3).cpu()

    # assumes mG is [row,cols,2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    return torch.from_numpy(mGrgb).permute(2, 0, 1)


def rescale_img(x, mode='scale', perc=None, tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            y = x.clone()
            xmin = np.percentile(y.detach().cpu().numpy(), perc)
            xmax = np.percentile(y.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = torch.clamp(x, 0, 1)
    return x


def to_uint8(x):
    return (255. * x).astype(np.uint8)


def to_numpy(x):
    return x.detach().cpu().numpy()


def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)).float()


class Camera(Dataset):
    def __init__(self, downsample_factor=1):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.img = Image.fromarray(skimage.data.camera())
        self.img_channels = 1

        if downsample_factor > 1:
            size = (int(512 / downsample_factor),) * 2
            self.img_downsampled = self.img.resize(size, Image.ANTIALIAS)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.downsample_factor > 1:
            return self.img_downsampled
        else:
            return self.img

import cv2
def func(im):
    # im needs to be 0-1
    # print(im.shape, im.max())
    if im.max() > 1:
        im = im / 255.0
    laplacian = cv2.Laplacian(im, cv2.CV_64F) 
    new = laplacian + 6 * im 
    ww = (new - new.min()) / (new.max() - new.min()) 
    return ww


def rotate(src, angle):
    rows,cols = src.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    dst = cv2.warpAffine(src, M, (cols,rows))
    return dst


class NoisyCamera(Dataset):
    def __init__(self, downsample_factor=1, noise_level=0, img_path='camera', sz=512, type=None):
        super().__init__()
        self.downsample_factor = downsample_factor
        if img_path == 'camera':
            self.img = (skimage.data.camera())
        else:
            self.img = (Image.open(img_path).convert('RGB').resize((sz, sz), Image.LANCZOS))

        if type == 'inpainting':
            # inpainting holes
            # for text inpainting, we add onto images and load as is
            mask = np.random.uniform(size=(256, 256, 1)) > 0.7
            # mask = np.random.uniform(size=(256, 256, 1)) > 0.85
            self.img = np.array(self.img)
            self.img = self.img * (1 - mask)
            self.img = Image.fromarray(self.img.astype(np.uint8))
        elif type == 'blur':
            img = np.array(self.img) / 255
            for i in range(5):
                img = func(img)
            self.img = Image.fromarray((img * 255).astype(np.uint8))
        elif type == 'noise':
            self.img = np.array(self.img)
            self.img = (self.img) / 255.0
            ww = noise_level
            gaussian = np.random.normal(0, ww / 255.0, (sz, sz, 3))
            self.img += gaussian
            self.img = self.img * 255
            self.img = np.clip(self.img, a_min=0, a_max=255).astype(np.uint8)
            self.img = Image.fromarray(self.img)

        self.img_channels = 3

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.downsample_factor > 1:
            return self.img_downsampled
        else:
            return self.img

class NoisyCamera_multimlp_rays(Dataset):
    def __init__(self, downsample_factor=1, noise_level=0, img_path='camera', target='sobel', img_num=100):
        super().__init__()
        self.img_channels = 3
        self.downsample_factor = downsample_factor
        if img_path == 'div2k':
            div2k_folder = './data/DIV2K_train_HR'
            self.li = glob.glob(div2k_folder + '/*.png')
            # img_num = 100
            self.li = sorted(self.li)[:img_num]
        else:
            self.li = [img_path]
        self.img_list = []
        for idx, img_path in enumerate(self.li):
            self.img = (Image.open(img_path).resize((512, 512), Image.LANCZOS))
            assert noise_level == 0
            self.img_list.append(self.img)
        self.len = len(self.img_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.img_list[idx], self.li[idx]

import glob2

class ImageFile(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.img = Image.open(filename)
        self.img_channels = len(self.img.mode)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img


class ImplicitAudioWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.grid = np.linspace(start=-100, stop=100, num=dataset.file_length)
        self.grid = self.grid.astype(np.float32)
        self.grid = torch.Tensor(self.grid).view(-1, 1)

    def get_num_samples(self):
        return self.grid.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        rate, data = self.dataset[idx]
        scale = np.max(np.abs(data))
        data = (data / scale)
        gaussian = np.random.normal(0, 25 / 255.0, data.shape)
        data += gaussian
        data = torch.Tensor(data).view(-1, 1)
        # data += torch.rand.uniform
        return {'idx': idx, 'coords': self.grid}, {'func': data, 'rate': rate, 'scale': scale}



class ImplicitAudioWrapper_ray(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.grid = np.linspace(start=-100, stop=100, num=dataset.file_length)
        self.grid = self.grid.astype(np.float32)
        self.grid = torch.Tensor(self.grid).view(-1, 1)

    def get_num_samples(self):
        return self.grid.shape[0]

    def __len__(self):
        return self.grid.shape[0]

    def __getitem__(self, idx):
        rate, data = self.dataset[idx]
        scale = np.max(np.abs(data))
        data = (data / scale)
        data = torch.Tensor(data).view(-1, 1)
        return {'idx': idx, 'coords': self.grid[idx]}, {'func': data[idx], 'rate': rate, 'scale': scale}

class AudioDenoise(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.grid = np.linspace(start=-100, stop=100, num=dataset.file_length)
        self.grid = self.grid.astype(np.float32)
        self.grid = torch.Tensor(self.grid).view(-1, 1)
        rate, data = self.dataset[0]
        scale = np.max(np.abs(data))
        data = (data / scale)
        self.gt = torch.Tensor(data).view(-1, 1)
        grad = np.load('./data/audio_grad/audio_noisy_10.npy')
        for i in range(grad.shape[0]):
            while (grad[i].max() > 10):
                grad[i] /= 256
        self.grad = torch.from_numpy(grad)

    def get_num_samples(self):
        return self.grid.shape[0]

    def __len__(self):
        return self.grid.shape[0]

    def __getitem__(self, idx):
        in_dict = {'idx': idx, 'coords': self.grid[idx], 'grad': self.grad[..., idx]}
        gt_dict = {'img': self.gt[idx]}
        return in_dict, gt_dict

class AudioFile(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.rate, self.data = wavfile.read(filename)
        if len(self.data.shape) > 1 and self.data.shape[1] == 2:
            self.data = np.mean(self.data, axis=1)
        self.data = self.data.astype(np.float32)
        self.file_length = len(self.data)
        print("Rate: %d" % self.rate)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.rate, self.data

class SDFWrapper_ray(torch.utils.data.Dataset):
    def __init__(self):
        self.mgrid = get_mgrid(256, 3)
        data = np.load('./data/sdf/siren_thai.npy').reshape(256, 256, 256)

        img = torch.from_numpy(data).float()
        self.gt = img.reshape(-1, 1)
        self.inp = self.mgrid
        print(self.mgrid.min(), self.mgrid.max())
        print(self.mgrid.shape)
    def __len__(self):
        return self.mgrid.shape[0]
    def __getitem__(self, idx):
        in_dict = {'idx': idx, 'coords': self.inp[idx]}
        gt_dict = {'img': self.gt[idx]}
        return in_dict, gt_dict

from scipy import ndimage
class SDFSmoothen_ray_uniform(torch.utils.data.Dataset):
    def __init__(self, sz=64, ti=1):
        self.mgrid = get_mgrid(sz, 3)
        name = 'armadillo'
        ww = (np.load(f'./data/sdf/{name}_ori.npy')).reshape(256, 256, 256)
        kk = np.zeros((3, 3, 3))
        kk[:, :, 1] = 1
        kk[:, 1, :] = 1
        kk[1][1][1] = 2
        kk /= np.sum(kk)
        for i in range(ti):
            ww = ndimage.convolve(ww, kk)
        img = torch.from_numpy(ww).float()
        self.gt = img.reshape(-1, 1)
        self.inp = self.mgrid
        grad = np.load(f'./data/sdf_grad/{name}_siren.npy')
        print(grad.shape) # 22, 262144
        for i in range(grad.shape[0]):
            # print(grad[i].max())
            while (grad[i].max() > 10):
                grad[i] /= 256
        self.grad = (torch.from_numpy(grad))
    def __len__(self):
        return 64
    def __getitem__(self, ori_idx):
        idx = []
        for i in range(64):
            for j in range(64):
                for k in range(64):
                    offset = np.random.uniform(0, 3, 3)
                    idx.append((i * 4 + offset[0]) * 256 * 256 + (j * 4 + offset[1]) * 256 + (k * 4 + offset[2]))
        idx = np.array(idx)
        in_dict = {'idx': idx, 'coords': self.inp[idx], 'grad': self.grad[..., idx].permute(1, 0)}
        # print(in_dict['grad'].shape)
        gt_dict = {'img': self.gt[idx]}
        return in_dict, gt_dict

class Implicit2DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, compute_diff=None, ti=1):

        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)
        self.sidelength = sidelength

        # print(sidelength)
        self.transform = Compose([
            # Resize(sidelength),
            ToTensor(),
            # Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.compute_diff = compute_diff
        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength)
        # self.mgrid /= self.mg
        print('max of grid', self.mgrid.max())
        self.len = len(self.dataset)
        self.ww = ti
    def __len__(self):
        # return len(self.dataset)
        return self.len * self.ww

    def __getitem__(self, idx):
        # print(self.dataset[idx % self.len])
        img = self.transform(self.dataset[idx % self.len])
        # print(img.shape)

        if self.compute_diff == 'gradients':
            # img *= 1e1
            gradx = scipy.ndimage.sobel(img[0].numpy(), axis=0)[..., None]
            grady = scipy.ndimage.sobel(img[0].numpy(), axis=1)[..., None]
            # gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            # grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
        elif self.compute_diff == 'laplacian':
            # img *= 1e4
            laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
        elif self.compute_diff == 'all':
            # print(scipy.ndimage.sobel(img.numpy(), axis=1).shape)
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
            laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]

        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        # offset = ((torch.rand(self.mgrid.shape) - 0.5) * 2) / 256
        # print(self.mgrid.max())
        # in_dict = {'idx': idx, 'coords': self.mgrid}
        in_dict = {'idx': idx, 'coords': self.mgrid}
        gt_dict = {'img': img}

        if self.compute_diff == 'gradients':
            gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
                                   torch.from_numpy(grady).reshape(-1, 1)),
                                  dim=-1)
            gt_dict.update({'gradients': gradients})

        elif self.compute_diff == 'laplacian':
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        elif self.compute_diff == 'all':
            # gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
            #                        torch.from_numpy(grady).reshape(-1, 1)),
            #                       dim=-1)
            gradients = torch.sqrt(torch.from_numpy(gradx).reshape(-1, 1) ** 2 + torch.from_numpy(grady).reshape(-1, 1) ** 2)
            gt_dict.update({'gradients': gradients})
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})
        else:
            pass

        return in_dict, gt_dict

    def get_item_small(self, idx):
        img = self.transform(self.dataset[idx])
        spatial_img = img.clone()
        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        gt_dict = {'img': img}

        return spatial_img, img, gt_dict

class Implicit2DWrapper_multimlp_ray(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, compute_diff=None, sigma=1):

        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)
        self.sidelength = sidelength
        self.sigma = sigma
        print('sz', sidelength)

        self.transform = Compose([
            Resize(sidelength),
            ToTensor(),
            # Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.compute_diff = compute_diff
        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength)
        # self.mgrid /= self.mg
        print('max of grid', self.mgrid.max())
        self.len = len(self.dataset) * sidelength[0] * sidelength[0]
        num = len(self.dataset)
        self.names = self.dataset.li
        rgb = []
        new_rgb = []
        coords = []
        for idx in range(num):
            im, name = self.dataset[idx]
            img = self.transform(im)
            img /= 256
            if self.compute_diff == 'gradients':
                img *= 1e1
                gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
                grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
            elif self.compute_diff == 'laplacian':
                img *= 1e4
                laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
            elif self.compute_diff == 'sobel':
                # print('???')
                gradx = scipy.ndimage.sobel((img.numpy()), axis=1).squeeze(0)[..., None]
                grady = scipy.ndimage.sobel((img.numpy()), axis=2).squeeze(0)[..., None]
                laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
            elif self.compute_diff == 'blur_x' or self.compute_diff == 'blur_y':
                blurx = scipy.ndimage.gaussian_filter1d((img.numpy()), sigma=self.sigma, axis=1)

            img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

            if self.compute_diff == 'gradients':
                gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
                                    torch.from_numpy(grady).reshape(-1, 1)),
                                    dim=-1)
                # gt_dict.update({'gradients': gradients})
                target = gradients

            elif self.compute_diff == 'laplacian':
                target = torch.from_numpy(laplace).view(-1, 1)
            elif self.compute_diff == 'sobel':
                gradients = torch.sqrt(torch.from_numpy(gradx).reshape(-1, 1) ** 2 + torch.from_numpy(grady).reshape(-1, 1) ** 2)
                target = gradients
            elif self.compute_diff == 'blur_x':
                target = torch.from_numpy(blurx).reshape(3, -1).permute(1, 0)
            elif self.compute_diff == 'blur_y':
                blury = torch.from_numpy(blury).reshape(-1, 1)
                target = blury
            else:
                raise NotImplementedError
            rgb.append(img)
            new_rgb.append(target)
            coords.append(self.mgrid.view(-1, 2))
        self.rgb = torch.cat(rgb, 0)
        self.new_rgb = torch.cat(new_rgb, 0)
        self.coords = torch.cat(coords, 0)
        print(self.rgb.shape, self.new_rgb.shape, self.coords.shape)
        self.sidelength = sidelength[0]

    def __len__(self):
        # return len(self.dataset)
        return self.len

    def __getitem__(self, idx):
        # im, name = self.dataset[idx]
        # img = self.transform(im)
        # img /= 256
        name = self.names[idx // self.sidelength // self.sidelength]

        

        in_dict = {'idx': idx, 'coords': self.coords[idx]}
        gt_dict = {'img': self.new_rgb[idx]}
        return in_dict, gt_dict

from skimage.feature import hog
def get_hog(im):
    # print(im.shape)
    return hog(im, visualize=True, multichannel=True)[1]

import tqdm
class Implicit2DWrapper_multimlp_ray_offline(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, target=None, sigma=1, split='train'):

        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)
        self.sidelength = sidelength
        self.sigma = sigma

        self.transform = Compose([
            Resize(sidelength),
            ToTensor(),
            # Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.target = target
        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength)
        # self.mgrid /= self.mg
        print('max of grid', self.mgrid.max())
        self.len = len(self.dataset) * sidelength[0] * sidelength[0]
        num = len(self.dataset)
        self.names = self.dataset.li
        rgb = []
        new_rgb = []
        input_grad = []
        coords = []
        
        if target == 'blur':
            base = './data/train_color_ori/'
            suff = '_ori'
        elif target == 'deblur':
            base = './data/train_color_blur/'
            suff = '_blur'
        elif target == 'denoise':
            base = './data/train_color_noise/'
            suff = '_noise'
        elif target == 'inpainting':
            base = './data/train_color_hole/'
            suff = '_hole'
        elif target == 'inpainting_text':
            base = './data/train_color_inpainting_text/'
            suff = '_inpainting_text'
            
        for idx in tqdm.tqdm(range(num)):
            im, name = self.dataset[idx]
            grad = np.load(os.path.join(base, 'div2k_' + os.path.basename(name).replace('.png', f'.npy_color{suff}.npy')))
            grad = grad.transpose(1, 0)
            for i in range(grad.shape[0]):
                # print(grad[i].max())
                while (grad[i].max() > 10):
                    grad[i] /= 256
            input_grad.append(torch.from_numpy(grad))
            img = self.transform(im)
            # img /= 256
            if self.target in ['blur']:
                blurx = func(func(img.permute(1, 2, 0).numpy().astype(np.float64)))
                for i in range(5):
                    blurx = func(blurx)
                target = torch.from_numpy(blurx).reshape(-1, 3)
                # target = blurx
            elif  self.target in ['deblur', 'denoise', 'inpainting', 'inpainting_text']:
                # use gt as training target
                target = (img).reshape(3, -1).permute(1, 0)
            elif self.target == 'sobel':
                target = sobel(img.unsqueeze(0), normalized=False).view(3, -1).permute(1, 0)
                # print(target.shape)
                # target = target.clamp(0, 1)
                # print(target.min(), target.max())
            elif self.target == 'hog':
                # print(im.shape, target.shape)
                hog = get_hog(np.array(im) / 255.0)
                # print(hog.shape) # 151, 201
                target = torch.from_numpy(hog).float().view(-1, 1)
            else:
                raise NotImplementedError
            img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)
            rgb.append(img)
            new_rgb.append(target)
            coords.append(self.mgrid.view(-1, 2))
        self.rgb = torch.cat(rgb, 0)
        self.new_rgb = torch.cat(new_rgb, 0)
        self.input_grad = torch.cat(input_grad, 1)
        self.coords = torch.cat(coords, 0)
        print(self.rgb.shape, self.new_rgb.shape, self.coords.shape, self.input_grad.shape, self.len)
        self.sidelength = sidelength[0]

    def __len__(self):
        # return len(self.dataset)
        return self.len

    def __getitem__(self, idx):
        in_dict = {'idx': idx, 'coords': self.coords[idx], 'grad': self.input_grad[..., idx]}
        gt_dict = {'img': self.new_rgb[idx]}
        return in_dict, gt_dict
