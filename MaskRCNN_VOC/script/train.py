import torch.multiprocessing as mp
import argparse

from TorchFramework import *
from VOC2012_MaskRCNN_InstanceSegmentation import *
import numpy as np
import sys

torch_tasks = [
    MaskRCNN(),
    MaskRCNN_Boundary_Reinforcement(boundary_size=1),
    MaskRCNN_Boundary_Reinforcement(boundary_size=2),
    MaskRCNN_Boundary_Reinforcement(boundary_size=3),
    MaskRCNN_Mask_Expending(expending_size=1),
    MaskRCNN_Mask_Expending(expending_size=2),
    MaskRCNN_Mask_Expending(expending_size=3),
]
save_dir_list = [
    "./result/2/benchmark",
    "./result/2/boundary(size=1)",
    "./result/2/boundary(size=2)",
    "./result/2/boundary(size=3)",
    "./result/2/expending(size=1)",
    "./result/2/expending(size=2)",
    "./result/2/expending(size=3)",
]

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("task_num", help= "몇 번째 테스크를 수행할 지", type=int)
    args = parser.parse_args()
    np.random.seed(2)

    i = args.task_num
    task_worker = TorchTaskWorker(
        task = torch_tasks[i], 
        epoch = 100, 
        batch_size = 5, 
        save_dir = save_dir_list[i], 
        device = i)
    task_worker.single_train_val_worker()