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

    def get_dataset_and_loader(self, world_size, rank, batch_size):
        raise NotImplementedError()

    def get_optimizer(self, parameters):
        raise NotImplementedError()
    
    def get_scheduler(self, optimizer):
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

    def to_for_batch(self, batch, device):
        raise NotImplementedError()

    def feed(self, model, batch):
        raise NotImplementedError()

    def result_to_loss(self, result):
        raise NotImplementedError()

class TorchTaskWorker(EnforceOverrides):
    def __init__(self, task):
        self.task = task

    def is_main(self, world_size, rank, main_rank):
        return (world_size == 1) or (1 < world_size and rank == main_rank)
    
    def train_one_epoch(self, model, dataloader, optimizer, scheduler, device= "cpu", 
                        show_log = False,
                        show_log_frequency = 1, 
                        show_progress = False,
        ):
        self.task.before_train_one_epoch(model)
        model.to(device)
        loss_list = []

        if show_progress:
            dataloader = tqdm(dataloader)
            
        for i, batch in enumerate(dataloader):
            
            batch = self.task.to_for_batch(batch, device)
            result = self.task.feed(model, batch)
            # if torch.distributed.get_rank() == 0:
            #     # print(result["loss_mask"])
            #     print(result)
            loss = self.task.result_to_loss(result)
            
            # back propergation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss)

            if show_log and (i+1) % show_log_frequency == 0:
                print(f"Loss: {loss}")
        
        scheduler.step()
        self.task.after_train_one_epoch(model)

        return torch.mean(torch.stack(loss_list)).item()
    
    def loss_eval(self, model, dataloader, device = "cpu",
                show_log = False, 
                show_log_frequency = 1, 
                show_progress = False,
        ):
        self.task.before_loss_eval(model)
        model.to(device)

        loss_list = []

        if show_progress:
            dataloader = tqdm(dataloader)

        for batch in dataloader:
            batch = self.task.to_for_batch(batch, device)
            result = self.task.feed(model, batch)
            loss = self.task.result_to_loss(result).detach().to("cpu")
            loss_list.append(loss)
            
            if show_log and (i+1) % show_log_frequency == 0:
                print(f"Loss: {loss}")

        self.task.after_loss_eval(model)
        return torch.mean(torch.stack(loss_list)).item()

    def train_val(self, model, train_dataloader, val_dataloader, optimizer, scheduler, epoch = 1, device = "cpu",
        val_frequency = 1, 
        show_log = False,
        show_log_frequency = 1,
        save_dir = "",
        save_checkpoint = True,
        save_checkpoint_frequency = 10,
        save_log = True, 
        save_model = True,
        
        #train
        train_show_log = False, 
        train_show_log_frequency = 1, 
        train_show_progress = False,
        
        #val
        val_show_log = False, 
        val_show_log_frequency = 1, 
        val_show_progress = False,
        ):
        """
        [Operation]
            입력된 모델과 데이터에 대해 정해진 에폭만큼 train, val을 수행하고 결과 반환
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        train_loss_list, val_loss_list = [], []

        for e in range(1, epoch+1):
            # train_loss 계산 및 학습
            train_loss = self.train_one_epoch(model, train_dataloader, optimizer, scheduler, device, 
                                                show_log = train_show_log, 
                                                show_log_frequency = train_show_log_frequency, 
                                                show_progress = train_show_progress)
            train_loss_list.append(train_loss)

            # val_frequency 마다 한 번씩 val_loss 계산
            if e % val_frequency == 0:
                val_loss = self.loss_eval(model, val_dataloader, device,
                                            show_log = val_show_log, 
                                            show_log_frequency = val_show_log_frequency, 
                                            show_progress = val_show_progress,
                                            )
            else:
                val_loss = None    
            val_loss_list.append(val_loss)


            # Print Log
            if (show_log) and (e % show_log_frequency == 0):
                print(f"Epoch({e}/{epoch}) T_loss: {train_loss} V_loss: {val_loss}")

            # Save Checkpoint
            if (save_checkpoint) and (e%save_checkpoint_frequency == 0):
                torch.save(model.state_dict(), f"{save_dir}/checkpoint({e})")
                if isinstance(model, DDP):
                    torch.save(model.module.state_dict(), f"{save_dir}/checkpoint({e})")
                else:
                    torch.save(model.state_dict(), f"{save_dir}/checkpoint({e})")

            # Save Log
            loss_dict = {"train_loss_list": train_loss_list, "val_loss_list": val_loss_list}
            if save_log:
                with open(f"{save_dir}/loss_dict.p", 'wb') as file:
                    pickle.dump(loss_dict, file)

        if save_model:
            if isinstance(model, DDP):
                torch.save(model.module.state_dict(), f"{save_dir}/final_model({epoch})")
            else:
                torch.save(model.state_dict(), f"{save_dir}/final_model({epoch})")

    # DDP

    def train_val_worker(self, rank = 0, world_size = 1, main_rank=0, batch_size =1, epoch = 1, device = "cpu",
            val_frequency = 1, 
            show_log = False,
            show_log_frequency = 1,
            save_dir = "",
            save_checkpoint = True,
            save_checkpoint_frequency = 10,
            save_log = True, 
            save_model = True,
            
            #train
            train_show_log = False, 
            train_show_log_frequency = 1, 
            train_show_progress = False,
            
            #val
            val_show_log = False, 
            val_show_log_frequency = 1, 
            val_show_progress = False,
        ):
        
        # 필요한 부품
        model = DDP(self.task.get_model().to(rank), device_ids=[rank])
        
        t_dataset, t_dataloader, v_dataset, v_dataloader = self.task.get_dataset_and_loader(world_size, rank, batch_size)

        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = self.task.get_optimizer(parameters)
        scheduler = self.task.get_scheduler(optimizer)

        # 메인 작업
        if self.is_main(world_size, rank, main_rank):
            self.train_val(model, t_dataloader, v_dataloader, optimizer, scheduler, epoch, rank, 
                val_frequency = val_frequency, 
                show_log = show_log,
                show_log_frequency = show_log_frequency,
                save_dir = save_dir,
                save_checkpoint = save_checkpoint,
                save_checkpoint_frequency = save_checkpoint_frequency,
                save_log = save_log, 
                save_model = save_model,
                
                #train
                train_show_log = train_show_log, 
                train_show_log_frequency = train_show_log_frequency, 
                train_show_progress = train_show_progress,
                
                #val
                val_show_log = val_show_log, 
                val_show_log_frequency = val_show_log_frequency, 
                val_show_progress = val_show_progress,
            )
        else:
            self.train_val(model, t_dataloader, v_dataloader, optimizer, scheduler, epoch, rank,  
                val_frequency = val_frequency, 
                show_log = False,
                show_log_frequency = show_log_frequency,
                save_dir = save_dir,
                save_checkpoint = False,
                save_checkpoint_frequency = save_checkpoint_frequency,
                save_log = False, 
                save_model = False,
                
                #train
                train_show_log = False, 
                train_show_log_frequency = train_show_log_frequency, 
                train_show_progress = False,
                
                #val
                val_show_log = False, 
                val_show_log_frequency = val_show_log_frequency, 
                val_show_progress = False,
            )



 

        