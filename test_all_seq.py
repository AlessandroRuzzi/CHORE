# run all seq.s 
import shutil
import os
import subprocess as sp
import time

idxcard = 4

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

behave_dir = "/data/xiwang/behave/sequences"
for seq in os.listdir(behave_dir):
    if seq.startswith("Date03"):
        for cat in ["suitcase", "backpack", "basketball"]:
            if cat in seq:
                while get_gpu_memory()[idxcard] < 50000:
                    time.sleep(60)
                print('CUDA_VISIBLE_DEVICES={} python recon/recon_fit_behave.py chore-release --save_name chore-release -s {}/{} &'.format(idxcard, behave_dir, seq))
                os.system('CUDA_VISIBLE_DEVICES={} python recon/recon_fit_behave.py chore-release --save_name chore-release -s {}/{} &'.format(idxcard, behave_dir, seq))
                time.sleep(60)

print("all test seq. done")