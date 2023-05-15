import torch

class DummyDataset(torch.utils.data.dataset.Dataset):
    def __len__(self):
        return 10
    
    def __getitem__(self, idx):
        return idx
    
dataset = DummyDataset()
torch.distributed.init_process_group("gloo", rank=0, world_size=1)
torch.utils.data.distributed.DistributedSampler(dataset)