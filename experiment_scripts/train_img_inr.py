# INRNet version
# Enable import from parent package
import torch
# torch.set_default_dtype(torch.float64)
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
from functools import partial
import inrnet


p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--clip_grad', type=float, default=1)
# p.add_argument('--batch_size', type=int, default=100)
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--noise_level', type=int, default=0)
p.add_argument('--sz', type=int, default=256)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=25,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--hidden_features', type=int, default=256)
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')

p.add_argument('--img_path', type=str, help='Path to Image.', default='camera')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()

img_dataset = dataio.NoisyCamera(noise_level=opt.noise_level, img_path=opt.img_path, sz=opt.sz)
coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=opt.sz, compute_diff='gradients', ti=opt.num_epochs)
image_resolution = (opt.sz, opt.sz)

dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=False, num_workers=4)

model = inrnet.INRNet(out_features=3)
model.cuda()

root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Define the loss
loss_fn = partial(loss_functions.image_mse_lip, None)
summary_fn = partial(utils.write_image_summary, image_resolution)

training.train(model=model, train_dataloader=dataloader, epochs=1, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, clip_grad=opt.clip_grad)
