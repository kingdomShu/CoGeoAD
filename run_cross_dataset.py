import os
import subprocess

# clip cache
os.environ["CLIP_CACHE_DIR"] = "/home/wxl/.cache/clip"
os.makedirs(os.environ["CLIP_CACHE_DIR"], exist_ok=True)

# ================= parameters =================
device = 0
mode = 0 # 0:train，1:test
epoch = 20
chosen_data = 0 # 0:mvtec3d->eyecandies, 1:eyecandies->mvtec3d
image_size = "336"
point_size = "336"
num_views = 9

features_list = list(range(21, 25))

if chosen_data == 0:
    train_dataset = "mvtec3d"
    train_data_path = "/home/wxl/data/mvtec-re"
    test_dataset = "eyecandies"
    test_data_path = "/home/wxl/data/Eyecandies-re/"
else:
    train_dataset = "eyecandies"
    train_data_path = "/home/wxl/data/Eyecandies-re/"
    test_dataset = "mvtec3d"
    test_data_path = "/home/wxl/data/mvtec-re/"

# 4. prompt
depth = [9]
n_ctx = [12]
t_n_ctx = [4]


for i in range(len(depth)):
    for j in range(len(n_ctx)):
        
        base_dir = f"{train_dataset}_{num_views}"
        save_dir = f"./exps_{base_dir}/"
        
        
        os.makedirs(save_dir, exist_ok=True)
        common_args = [
            "--features_list"] + [str(x) for x in features_list] + [
            "--image_size", image_size,
            "--point_size", point_size,
            "--depth", str(depth[i]),
            "--t_n_ctx", str(t_n_ctx[0]),
            "--num_views", str(num_views),
            "--use_view_attention", str(1),
            "--feature_type", "both",
            "--view_attention_type", "ViewAttention",
            "--use_weighted_sum", str(1)  
        ]

        if mode == 0:
            # train
            print(f"Running training mode: {base_dir}")
            cmd = [
                "python", "train.py",
                "--dataset", train_dataset,
                "--train_data_path", train_data_path,
                "--save_path", save_dir,
                "--batch_size", "4",
                "--print_freq", "1",
                "--epoch",str(epoch),
                "--save_freq", "1",
                "--n_ctx", str(n_ctx[j]),
            ] + common_args
            
        elif mode == 1:
            # test
            print(f"Running testing mode: {base_dir}")
            cmd = [
                "python", "test.py",
                "--dataset", test_dataset,
                "--data_path", test_data_path,
                "--save_path", f"./exps_{base_dir}/",
                "--checkpoint_path", f"{save_dir}epoch_"+ str(epoch) +".pth",
                "--n_ctx", str(n_ctx[j]),  
            ] + common_args
            
        else:
            print("Invalid mode. Use 0 for training or 1 for testing.")
            exit(1)
        

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        
        try:
            print(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env, check=True)
            print(f"Command completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")
            exit(1)