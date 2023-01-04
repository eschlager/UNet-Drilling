"""
Script for running evaluate_model on all models in a folder

@author: eschlager
January 2023
"""

import multiprocessing as mp
import os
import random
import time

import evaluate_cv

random.seed(42)


def run_evaluation(gpu_id, model_folder, data_folder):
    evaluate_cv.main(gpu_id, model_folder, data_folder)


if __name__ == "__main__":
    start_time = time.time()

    GPU_IDS = [0, 1]

    path_models = os.path.join(os.path.abspath(os.pardir), 'models')
    model_folders = os.listdir(path_models)

    i = 0
    ps_per_gpus = dict.fromkeys(GPU_IDS)
    while i < len(model_folders):
        # detect if any GPU is not busy
        if (not all(ps_per_gpus.values())) or (not all([p.is_alive() for p in ps_per_gpus.values()])):
            print("Not all GPUs busy!")
            for gpu_id in GPU_IDS:  # iterate through the GPUs and get the not busy ones a job:
                if ((ps_per_gpus[gpu_id] is None) or (not ps_per_gpus[gpu_id].is_alive())) and (i < len(model_folders)):
                    if '256' in model_folders[i]:
                        data_folder = 'data/images/interim/dev_centertiles_256'
                    else:
                        data_folder = 'data/images/interim/dev_centertiles_512'
                    ps_per_gpus[gpu_id] = mp.Process(target=run_evaluation, args=(
                        gpu_id, os.path.join('output/image_segmentation/', model_folders[i]), data_folder))
                    print(f"Start Training of model {i} on GPU core {gpu_id}.")
                    ps_per_gpus[gpu_id].start()
                    i += 1

    print(f"Finished evaluation!")
