import os, glob

div2k_folder = './data/mnist/train/'
GPU_NUM = 6
gpu_ = [0, 1, 2, 3, 6, 7]
for cls in range(7, 10):
    li = glob.glob(div2k_folder + f'{cls}/*.png')
    li = sorted(li)
    
    for idx, cur in enumerate(li):
        if os.path.exists(f'logs/mnist_train_{cls}_{os.path.basename(cur)}/checkpoints/model_final.pth'):
            continue
        print(f"CUDA_VISIBLE_DEVICES={gpu_[idx % GPU_NUM]} python experiment_scripts/train_img.py --model_type=sine --experiment_name mnist_train_{cls}_{os.path.basename(cur)} --noise_level 0 --steps_til_summary 20000 --num_epochs 1001 --epochs_til_ckpt 1000 --img_path {cur} --hidden_features 32 --sz 28 &")
        if idx and idx % GPU_NUM == GPU_NUM - 1:
            print("sleep 8") # 5 min
