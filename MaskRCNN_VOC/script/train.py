import torch.multiprocessing as mp
import argparse

from TorchFramework import *
from VOC2012_MaskRCNN_InstanceSegmentation import *
import numpy as np
import sys
import math

from multiprocessing import Process
import os

torch_tasks = []

name = "14x14"
size = (14, 14)
torch_tasks.append(MaskRCNN(f"Benchmark({name})_v1", roi_pooling_size = size))
for i in range(1, 6):
    torch_tasks.append(MaskRCNN_Boundary_Reinforcement(f"Boundary(size={i})({name})_v1", roi_pooling_size = size, boundary_size=i))

name = "30x30"
size = (30, 30)
torch_tasks.append(MaskRCNN(f"Benchmark({name})_v1", roi_pooling_size = size))
for i in range(1, 6):
    torch_tasks.append(MaskRCNN_Boundary_Reinforcement(f"Boundary(size={i})({name})_v1", roi_pooling_size = size, boundary_size=i))

name = "50x50"
size = (50, 50)
torch_tasks.append(MaskRCNN(f"Benchmark({name})_v1", roi_pooling_size = size))
for i in range(1, 6):
    torch_tasks.append(MaskRCNN_Boundary_Reinforcement(f"Boundary(size={i})({name})_v1", roi_pooling_size = size, boundary_size=i))

worker_num = 7
# MaskRCNN_Flip(),
# MaskRCNN_Jitter(),
# MaskRCNN_Cropping(),
# MaskRCNN_C_F_J(),
# MaskRCNN_Boundary_Reinforcement(boundary_size=1),
# MaskRCNN_Boundary_Reinforcement(boundary_size=2),
# MaskRCNN_Boundary_Reinforcement(boundary_size=3),
# MaskRCNN_Mask_Expending(expending_size=1),
# MaskRCNN_Mask_Expending(expending_size=2),
# MaskRCNN_Mask_Expending(expending_size=3),


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_idx", help= "몇 번째 테스크를 수행할 지", type=int)
    args = parser.parse_args()
    np.random.seed(0)

    for task in torch_tasks:
        print(task.get_name())

    task_idx = args.task_idx
    device = args.task_idx
    while task_idx < len(torch_tasks):
        if 1 <= task_idx <= 5:
            print("Task: ", task_idx)
            def f():
                task_worker = TorchTaskWorker(
                    task = torch_tasks[task_idx], 
                    epoch = 50, 
                    batch_size = 5, 
                    device = device)
                task_worker.single_train_val_worker()

            p = Process(target = f, args=())
            p.start()
            p.join()
        task_idx += worker_num


