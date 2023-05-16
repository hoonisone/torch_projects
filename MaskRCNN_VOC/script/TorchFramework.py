import torch
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import pickle
from overrides import EnforceOverrides

class TorchTask(EnforceOverrides):
    # Element
    def get_model(self):
        raise NotImplementedError()

    def get_optimizer(self, parameters):
        raise NotImplementedError()
    
    def get_scheduler(self, optimizer):
        raise NotImplementedError()

    # dataset & dataloader
    def get_train_dataset(self):
        raise NotImplementedError()
    def get_val_dataset(self):
        raise NotImplementedError()
    def get_train_dataloader(self, dataset, batch_size = 1, world_size=1, rank=0):
        raise NotImplementedError()
    def get_val_dataloader(self, dataset, batch_size = 1, world_size=1, rank=0):
        raise NotImplementedError()

    # device
    def to_for_batch_data(self, batch_data, device):
        raise NotImplementedError()
    def to_for_sample_data(self, sample_data, device):
        raise NotImplementedError()
    def to_for_batch_result(self, batch_result, device):
        raise NotImplementedError()
    def to_for_sample_result(self, sample_result, device):
        raise NotImplementedError()

    # Context
    def before_train_one_epoch(self, model):
        model.train()

    def after_train_one_epoch(self, model):
        pass

    def before_loss_eval(self, model):
        model.eval()

    def after_loss_eval(self, model):
        pass

    def feed(self, model, batch):
        raise NotImplementedError()

    def result_to_loss(self, result):
        raise NotImplementedError()

class TorchTaskWorker(EnforceOverrides):
    def __init__(self, task, epoch=1, batch_size=1, device = "cpu", save_dir = ""):
        self.task = task
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device
        self.save_dir = save_dir


        # default options
        self.val_frequency = 1
        self.show_log = True
        self.show_log_frequency = 1
        self.save_checkpoint = True
        self.save_checkpoint_frequency = 10
        self.save_log = True
        self.save_model = True
        
        #train
        self.train_one_epoch_show_log = False
        self.train_one_epoch_show_log_frequency = 1,
        self.train_one_epoch_show_progress = True
        
        #val
        self.loss_eval_show_log = False
        self.loss_eval_show_log_frequency = 1
        self.loss_eval_show_progress = True

    def is_main(self, world_size, rank, main_rank):
        return (world_size == 1) or (1 < world_size and rank == main_rank)
    
    def train_one_epoch(self, model, dataloader, optimizer, scheduler, device= "cpu"):
        self.task.before_train_one_epoch(model)

        model.to(device)
        loss_list = []

        if self.train_one_epoch_show_progress:
            dataloader = tqdm(dataloader)
            
        for i, batch in enumerate(dataloader):
            batch = self.task.to_for_batch_data(batch, device)
            result = self.task.feed(model, batch)
            loss = self.task.result_to_loss(result)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss)

            if self.train_one_epoch_show_log and (i+1) % self.train_one_epoch_show_log_frequency == 0:
                print(f"Loss: {loss}")
        
        scheduler.step()
        self.task.after_train_one_epoch(model)

        return torch.mean(torch.stack(loss_list)).item()
    
    def loss_eval(self, model, dataloader, device = "cpu"):
        self.task.before_loss_eval(model)
        model.to(device)

        loss_list = []

        if self.loss_eval_show_progress:
            dataloader = tqdm(dataloader)

        for batch in dataloader:
            batch = self.task.to_for_batch_data(batch, device)
            result = self.task.feed(model, batch)
            loss = self.task.result_to_loss(result).detach().to("cpu")
            loss_list.append(loss)
            
            if self.loss_eval_show_log and (i+1) % self.loss_eval_show_log_frequency == 0:
                print(f"Loss: {loss}")

        self.task.after_loss_eval(model)
        return torch.mean(torch.stack(loss_list)).item()

    def train_val(self, model, train_dataloader, val_dataloader, optimizer, scheduler, epoch = 1, device = "cpu"):
        """
        [Operation]
            입력된 모델과 데이터에 대해 정해진 에폭만큼 train, val을 수행하고 결과 반환
        """
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        train_loss_list, val_loss_list = [], []

        for e in range(1, epoch+1):

            # distributed sampler의 경우 매 애폭마다 호출 필요
            if isinstance(model, DDP):
                dataloader.sampler.set_epoch(epoch)

            # train_loss 계산 및 학습
            train_loss = self.train_one_epoch(model, train_dataloader, optimizer, scheduler, device)
            train_loss_list.append(train_loss)

            # val_frequency 마다 한 번씩 val_loss 계산
            if e % self.val_frequency == 0:
                val_loss = self.loss_eval(model, val_dataloader, device)
            else:
                val_loss = None    
            val_loss_list.append(val_loss)

            # Print Log
            if (self.show_log) and (e % self.show_log_frequency == 0):
                print(f"Epoch({e}/{epoch}) T_loss: {train_loss} V_loss: {val_loss}")

            # Save Checkpoint
            if (self.save_checkpoint) and (e%self.save_checkpoint_frequency == 0):
                torch.save(model.state_dict(), f"{self.save_dir}/checkpoint({e})")
                if isinstance(model, DDP):
                    torch.save(model.module.state_dict(), f"{self.save_dir}/checkpoint({e})")
                else:
                    torch.save(model.state_dict(), f"{self.save_dir}/checkpoint({e})")

            # Save Log
            loss_dict = {"train_loss_list": train_loss_list, "val_loss_list": val_loss_list}
            if self.save_log:
                with open(f"{self.save_dir}/loss_dict.p", 'wb') as file:
                    pickle.dump(loss_dict, file)

        if self.save_model:
            if isinstance(model, DDP):
                torch.save(model.module.state_dict(), f"{self.save_dir}/final_model({epoch})")
            else:
                torch.save(model.state_dict(), f"{self.save_dir}/final_model({epoch})")


    def single_train_val_worker(self):
        model = self.task.get_model().to(self.device)
        t_dataloader = self.task.get_train_dataloader(self.task.get_train_dataset(), batch_size = self.batch_size)
        v_dataloader = self.task.get_val_dataloader(self.task.get_val_dataset(), batch_size = self.batch_size)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = self.task.get_optimizer(parameters)
        scheduler = self.task.get_scheduler(optimizer)
        self.train_val(model, t_dataloader, v_dataloader, optimizer, scheduler, epoch = self.epoch, device = self.device)

    # DDP
    # def parallel_train_val_worker(self, rank = 0, world_size = 1, main_rank=0, epoch = 1, device = "cpu"):
    #     # 필요한 부품
    #     model = DDP(self.task.get_model().to(device), device_ids=[device])
        
    #     t_dataloader = self.get_train_dataloader(self.get_train_dataset(), world_size, rank)
    #     v_dataloader = self.get_val_dataloader(self.get_val_dataset(), world_size, rank)
        
    #     parameters = [p for p in model.parameters() if p.requires_grad]
    #     optimizer = self.task.get_optimizer(parameters)
    #     scheduler = self.task.get_scheduler(optimizer)

    #     self.train_val(model, t_dataloader, v_dataloader, optimizer, scheduler, epoch, device) 

    def eval_and_save(task):
        model = self.task.get_model()
        dataloader = self.task.get_eval_dataloader()
        device = self.task.get_device()
        
        model.eval()
        model.to(device)
        
        idx = 0
        for batch in dataloader:
            batch = task.to_for_batch_data(batch, device)
            batch_result = model(batch)
            for sample_result in batch_result:
                idx += 1
                with open(f"{save_dir}/eval/sample_result({idx}).p", 'wb') as file:
                    pickle.dump(sample_result, file)
