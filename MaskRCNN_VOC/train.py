import torch.multiprocessing as mp

from TorchFramework import *
from VOC2012_MaskRCNN_InstanceSegmentation import *
import numpy as np


def setup(rank, world_size):
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()

def worker(rank, world_size, torch_task, save_dir):
    setup(rank, world_size)
    TorchTaskWorker(torch_task).train_val_worker(rank = rank, world_size = world_size, main_rank=0, batch_size = 4, epoch = 30, device = rank,
        # train val option
        val_frequency = 1, 
        show_log = True,
        show_log_frequency = 1,
        save_dir = save_dir,
        save_checkpoint = True,
        save_checkpoint_frequency = 1,
        save_log = True, 
        save_model = True,
        
        #train option
        train_show_log = False, 
        train_show_log_frequency = 1, 
        train_show_progress = True,
        
        #val option
        val_show_log = False, 
        val_show_log_frequency = 1, 
        val_show_progress = True,
    )
    cleanup()

if __name__=="__main__":
    world_size = 7
    print(f"""*****Start Multi Processing (PyTorch)*****\n* WorldSize: {world_size}\n""")

    torch_tasks = [
        # MaskRCNN(),
        # MaskRCNN_Boundary_Reinforcement(boundary_size=1),
        # MaskRCNN_Boundary_Reinforcement(boundary_size=2),
        # MaskRCNN_Boundary_Reinforcement(boundary_size=3),
        MaskRCNN_Mask_Expending(expending_size=1),
        MaskRCNN_Mask_Expending(expending_size=2),
        MaskRCNN_Mask_Expending(expending_size=3),
    ]
    save_dir_list = [
        # "./result/benchmark",
        # "./result/boundary(size=1)",
        # "./result/boundary(size=2)",
        # "./result/boundary(size=3)",
        "./result/expending(size=1)",
        "./result/expending(size=2)",
        "./result/expending(size=3)",
    ]

    for torch_task, save_dir in zip(torch_tasks, save_dir_list):
        np.random.seed(0)
        mp.spawn(worker,
            args=(world_size, torch_task, save_dir),
            nprocs=world_size,
            join=True)

    print("End")
