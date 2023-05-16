import torch.multiprocessing as mp

from TorchFramework import *
from VOC2012_MaskRCNN_InstanceSegmentation import *
import numpy as np

from PIL import Image 

from torch.nn.parallel import DistributedDataParallel as DDP


if __name__=="__main__":
    world_size = 1
    rank = 0

    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    device = 0
    model = Task.get_model().to(device)
    model.eval()
    model = DDP(model, device_ids=[0])
    model.load_state_dict(torch.load("./result/benchmark/checkpoint(15)"), strict = False)

    print(torch.load("./result/benchmark/checkpoint(15)"))
    # t_dataset, t_dataloader, v_dataset, v_dataloader = Task.get_dataset_and_loader(10, shuffle = False)

    # batch = next(iter(v_dataloader))
    # images, targets = Device.to_for_batch_data(batch, device)
    # result = model(images, targets)

    # mask1 = result[0]["masks"][2].detach().to("cpu").numpy().squeeze()
    # mask2 = result[1]["masks"][2].detach().to("cpu").numpy().squeeze()
    # mask3 = result[2]["masks"][2].detach().to("cpu").numpy().squeeze()

    # plt.imsave('mask1.png', mask1)
    # plt.imsave('mask2.png', mask2)
    # plt.imsave('mask3.png', mask3)

    # plt.imsave('image1.png', images[0].detach().to("cpu").numpy().transpose((1, 2, 0)))
    # plt.imsave('image2.png', images[1].detach().to("cpu").numpy().transpose((1, 2, 0)))
    # plt.imsave('image3.png', images[2].detach().to("cpu").numpy().transpose((1, 2, 0)))
    