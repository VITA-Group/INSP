import os, glob

div2k_folder = '/dataset/DIV2K_train_HR'
li = glob.glob(div2k_folder + '/*.png')
li = sorted(li)[:200]
GPU_NUM = 8
gpu_ = range(8)

for idx, cur in enumerate(li):
    print(f"CUDA_VISIBLE_DEVICES={gpu_[idx % GPU_NUM]} python experiment_scripts/train_img.py --model_type=sine --experiment_name div2k_{os.path.basename(cur)}_color_noise --noise_level 25 --steps_til_summary 20000 --num_epochs 10001 --epochs_til_ckpt 10000 --img_path {cur} --hidden_features 256 --sz 256 --lr 1e-4 &")
    if idx and idx % GPU_NUM == GPU_NUM - 1:
        print("sleep 125")
