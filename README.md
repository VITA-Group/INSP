# Signal Processing for Implicit Neural Representations

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

The official implementation of NeurIPS 2022 paper ["Signal Processing for Implicit Neural Representations"]().

Dejia Xu*, Peihao Wang*, Yifan Jiang, Zhiwen Fan, Zhangyang (Atlas) Wang

[[Paper]](https://arxiv.org/abs/2210.08772) [[Website]](https://vita-group.github.io/INSP)

## Method Overview

![](./docs/static/media/overview.e47f8ec0149b9912e940.png)

![](./docs/static/media/framework.0c59d0c8b8386b9f7f45.png)

## Environment

You can then set up a conda environment with all dependencies like so:

```
conda env create -f environment.yml
conda activate INSP
```

## High-Level structure

- Fit INR
- Export gradients for INR
- Train INSP-Net
- Inference INSP-Net

## Image Processing

For image processing, we experiment on div2k dataset.

```bash
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip &
unzip DIV2K_train_HR.zip
```

- Fit multiple INR 

    Use `--type ` to specify the type of images you want to train on.

    ```python gen_div2k.py | zsh```

- Export gradients for INR 

    `--load` is used for `glob` to filter out corresponding INRs.

    ```
    python export_colorray.py --save_dir grad/train_color_noise/ --load 'div2k*.png_color_noise_'
    ```

    Then, manually divide `grad/train_color_noise` and put a few of them into `grad/test_color_noise` (in our case we used first 100 images in DIV2K for training and the following 100 images for testing)

- Train INSP-Net

    `--img_num` changes the number of images that are used for training.
    
    The training should converge in a couple of minutes.

    ```
    python experiment_scripts/train_img_grad_offline.py --model_type=sine --experiment_name denoise --noise_level 0 --target denoise --img_num 100 --overwrite --sigma 7 --sz 256 --batch_size 10240 --lr 1e-4
    ```

- Inference INSP-Net

    ```
    python eval_insp.py --save_path output/denoise --target denoise --ckpt_path logs/denoise/checkpoints/model_current.pth
    ```

The INRs used in our experiments can be found [here](https://drive.google.com/drive/folders/1VaEgKiWIGpQhIw5uxPJGWL0OdTTM-cuo?usp=sharing).

## Audio Denoising

- Fit INR

    ```
    python experiment_scripts/train_audio.py --model_type=sine --wav_path=data/gt_bach.wav --experiment_name audio_noisy_10
    ```

- Export gradients for INR 

    ```
    python export_audio.py
    ```

- Train INSP-Net

    ```
    python experiment_scripts/train_audio_insp.py --experiment_name audio_denoise --batch_size 10240
    ```

- Inference INSP-Net

    ```
    python eval_audio_insp.py
    ```

## SDF Smoothing


- Fit INR

    ```
    
    ```

- Export gradients for INR 

    ```
    python export_sdf_ray.py
    ```

- Train INSP-Net

    ```
    python experiment_scripts/train_sdf_insp.py --experiment_name smooth_armadillo --sz 256 --ti 10 --batch_size 1
    ```

- Inference INSP-Net

    ```
    python eval_sdf_insp.py
    ```


## Image Classification

Due to the large size of MNIST and CIFAR INRs, we can't provide all of the checkpoints. However, we share the scripts to generate the INRs.

## Citation

```
@inproceedings{Xu_2022_INSP,
    title={Signal Processing for Implicit Neural Representations},
    author={Xu, Dejia and Wang, Peihao and Jiang, Yifan and Fan, Zhiwen and Wang, Zhangyang},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2022}
}
```

