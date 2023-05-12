import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import numpy as np
from pathlib import Path


from VOC2012_MaskRCNN_InstanceSegmentation import *

import pickle

import bbox_visualizer as bbv
import matplotlib.pyplot as plt



def train_val(model, train_data_loader, val_data_loader, optimizer, scheduler, epoch = 1, device = "cpu",
    val_frequence = 1, print_frequency = None, step_save_frequency =  None, save_dir = ""):
    """
    [Operation]
        입력된 모델과 데이터에 대해 정해진 에폭만큼 train, val을 수행하고 결과 반환
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    train_loss_list, val_loss_list = [], []

    for e in range(1, epoch+1):
        # train_loss 계산 및 학습
        train_loss = Task.train_one_epoch(model, train_data_loader, optimizer, scheduler, device)
        train_loss_list.append(train_loss)

        # val_frequence 마다 한 번씩 val_loss 계산
        val_loss = Task.loss_eval(model, device, val_data_loader) if (e % val_frequence == 0) else val_loss_list[-1]        
        val_loss_list.append(val_loss)

        # print_frequency마다 한번 씩 loss 출력
        if (print_frequency is not None) and (e % print_frequency == 0):
            print(f"epoch({e}) train_loss: {train_loss} val_loss: {val_loss}")

        if (step_save_frequency is not None) and (e%step_save_frequency == 0):
            torch.save(model.state_dict(), f"{save_dir}/step_model({e}-epoch)")

        
    loss_dict = {"train_loss_list": train_loss_list, "val_loss_list": val_loss_list}

    with open(f"{save_dir}/loss_dict.p", 'wb') as file:
        pickle.dump(loss_dict, file)


    torch.save(model.state_dict(), f"{save_dir}/final_model({epoch}-epoch)")



def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    # 작업 그룹 초기화
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()


def worker(rank = 0, world_size=1):
    print(rank, world_size)
    setup(rank, world_size)


    np.random.seed(0)

    device = rank if rank != None else "cpu"
    model = Task.get_model().to(rank)
    model = DDP(model, device_ids=[rank])
    t_dataset, t_dataloader, v_dataset, v_dataloader = Task.get_dataset_and_loader(3, rank=rank, world_size=world_size)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)


    train_val(model, t_dataloader, v_dataloader, optimizer = optimizer, scheduler = scheduler,
            epoch=200, device = device,
        val_frequence = 1, print_frequency = 10, step_save_frequency = 10, save_dir = "result/test")
    
    cleanup()


def main():
    print("start")
    world_size = 3
    
    mp.spawn(worker,
        args=(world_size,),
        nprocs=world_size,
        join=True)
    print("end")


if __name__=="__main__":
    main()