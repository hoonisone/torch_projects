import os
import math
import torch
import pickle
import torchvision

import numpy as np
import mh_utils as MH
import bbox_visualizer as bbv
import matplotlib.pyplot as plt
from overrides import overrides


from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Any
from pathlib import Path
from torchvision.datasets.voc import *
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from TorchFramework import *

from customize import torch_customize

torch_customize.customize_all()
    
from torchvision.transforms import Compose
VOC_LABEL = [   "background", 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus','car','cat','chair','cow',
                'diningtable', 'dog','horse','motorbike', 'person','pottedplant', 'sheep','sofa','train','tvmonitor']
VOC_LABEL_PAIR = {VOC_LABEL[i]:i for i in range(len(VOC_LABEL))}


WRONG_FILE_NAMES_IN_TRAIN = ['2009_005069']
WRONG_FILES_NAMES_IN_VAL = ['2008_005245', '2009_000455', '2009_004969', '2011_002644', '2011_002863']
WRONG_FILES_NAMES = WRONG_FILE_NAMES_IN_TRAIN+WRONG_FILES_NAMES_IN_VAL
def remove_wrong_annotated_file(file_names):
    for wrong_file_name in WRONG_FILES_NAMES:
        if wrong_file_name in file_names:
            file_names.remove(wrong_file_name)
    return file_names


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        image_set: str = "train",
        image_transform: Optional[Callable] = Compose([torchvision.transforms.ToTensor()]),
        label_transform: Optional[Callable] = None,
        box_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        cropping = False,
        mask_expanding = False,
        flip = False,
        jitter = False,
        expanding_size = 1
    ):

        root = Path(root)
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.box_transform = box_transform
        self.mask_transform = mask_transform
        self.cropping = cropping
        self.mask_expanding = mask_expanding
        self.flip = flip
        self.jitter = jitter
        self.expanding_size = expanding_size

        self.image_set = verify_str_arg(image_set, "image_set", ["train", "trainval", "val"])

        split_f = root/"ImageSets"/"Segmentation"/f"{image_set}.txt"
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]
            file_names = remove_wrong_annotated_file(file_names)
            
        # dir setting
        image_dir = root/"JPEGImages"
        annotation_dir = root/"Annotations"
        mask_dir = root/"SegmentationObject"

        # file name list setting
        self.images = [image_dir/f"{x}.jpg" for x in file_names]
        self.annotations = [annotation_dir/f"{x}.xml" for x in file_names]
        self.masks = [mask_dir/f"{x}.png" for x in file_names]
        
        assert len(self.images) == len(self.annotations) == len(self.masks)
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index:int):
        image = self.get_image(index)
        labels, boxes = self.get_annotation(index)
        masks = self.get_masks(index)
        
        labels = torch.tensor(labels)
        boxes = torch.tensor(boxes)
        masks = torch.tensor(masks)

        # random_cropping ###########################################
        if self.cropping:
            c, h, w = image.shape
            image, masks = MH.random_cropping(image, masks, size=(min(200, h), min(200, w)))
            # mask에 아직 객체의 mask가 남아있는가?
            is_instance = masks.reshape(masks.shape[0], -1).any(1) == True
            # indexes에 해당하는 것만 뽑아서 mask list로 형태 복원
            boxes = torch.stack([MH.extract_box_from_binary_mask(mask) for mask in masks])

            # boxes = torch.zeros(boxes.shape)
            # boxes = torch.tensor([[0, 0, 1, 1] for i in range(boxes.shape[0])])
            labels = labels * is_instance.type(torch.uint8)
        ###########################################

        # expend mask ###########################################
        # 각 마스크를 상하좌우 1픽셀식 확장
        if self.mask_expanding:
            masks = [torch.tensor(MH.expend_mask(mask.numpy(), self.expanding_size)) for mask in masks]
            masks = torch.stack(masks)
        ###########################################

        # Flip #########################################

        if self.flip:
            if np.random.rand() < 0.5:
                image = torchvision.transforms.RandomHorizontalFlip(p=1)(image)
                masks = torchvision.transforms.RandomHorizontalFlip(p=1)(masks)
            
            if np.random.rand() < 0.5:
                image = torchvision.transforms.RandomVerticalFlip(p=1)(image)
                masks = torchvision.transforms.RandomVerticalFlip(p=1)(masks)
            
        if self.jitter:
            image = torchvision.transforms.ColorJitter(brightness=(0.3, 3), contrast=(0.45, 0.55), saturation=(0.45, 0.55), hue=(0.45, 0.45))(image)
        target = {
            "labels": labels,
            "boxes" : boxes,
            "masks" : masks
        }
        
        return (image, target)

    
    def get_image(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        return image
    
    def get_masks(self, index):
        mask = Image.open(self.masks[index])
        mask = np.array(mask)

        id_list = np.unique(mask)
        id_list = np.delete(id_list, np.where((id_list == 0) | (id_list == 255)))

        masks = (mask[None, :] == id_list.reshape(-1, 1, 1)).astype(np.uint8)
    
        if self.mask_transform is not None:
            masks = self.image_transform(masks)

        return masks
    
    def get_annotation(self, index):# labels, boxes
        annotation = VOCDetection.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
        objects = annotation["annotation"]["object"]
        
        labels = np.array([VOC_LABEL_PAIR[o["name"]] for o in objects], dtype = np.int64)
        boxes = [o["bndbox"] for o in objects]
        boxes = np.array([[box["xmin"], box["ymin"], box["xmax"], box["ymax"]] for box in boxes], dtype = np.float32)

        if self.label_transform is not None:
            labels = self.label_transform(labels)
        
        if self.box_transform is not None:
            boxes = self.box_transform(boxes)
        
        return (labels, boxes)


class Task:

    


    @staticmethod
    def get_dataloader(dataset, batch_size = 1, world_size = 1, rank = 0, num_workers = 1, shuffle = True):
        def collate_batch(batch):
            images = [sample[0] for sample in batch]
            targets = [sample[1] for sample in batch]
            
            return images, targets
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas = world_size, rank = rank, shuffle = shuffle, seed = 0, drop_last = False)
        return torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, collate_fn=collate_batch, sampler = sampler)

    @staticmethod
    def show_comparison_for_one_sample(task, idx, device, iou_threshold = 0.5, mask_binary_threshold=0.5):
        """
        [Operator]
            단일 데이터(sample)에 대해 model을 통과 시키고 검출 결과와 정답 결과를 대응시킨뒤 이를 이미지 리스트 형식으로 출력한다.
            이때 검출 여부로 iou_threshold를 고려하며
            마스크를 바이너리로 변환 시  mask_binary_threshol를 고려한다.
        """
        model = task.get_model()
        _, _, v_dataset, v_dataloader = task.get_simple_dataset()
        sample = v_dataset[idx]

        sample_result = __class__.eval_sample(task, model, sample, device)
        sample_result = task.to_for_sample_result(sample_result, device)
        sample_result = Converter.tensor_to_numpy_for_sample_result(sample_result)
        sample_result["masks"] = MH.to_binary_by_threshold(sample_result["masks"], threshold = mask_binary_threshold)
        sample = Converter.tensor_to_numpy_for_sample_data(sample)
        candidates, targets, iou_list = Evaluator.compare_result_and_gt(sample_result, sample, 21, iou_threshold)

        Visualizer.show_comparison_result(candidates, targets, iou_list)

    @staticmethod
    def eval_sample(task, model, sample, device = "cpu"):
        model.eval()
        model.to(device)
        image, target = task.to_for_sample_data(sample, device)
        result = model([image], [target])[0]
        return  Converter.squeeze_dimention_for_sample_result(result)
        


    @staticmethod
    def filter_sample_by_score(sample_result, threshold):
        scores, labels, boxes, masks = __class__.dict_to_tuple_for_result(sample_result)
        indexes =  np.where(scores>threshold)[0]
        return __class__.tuple_to_dict_for_result(scores[indexes], labels[indexes], boxes[indexes], masks[indexes])

class Converter:

    # dimention
    @staticmethod
    def squeeze_dimention_for_sample_result(sample_result):
        sample_result["masks"] = torch.squeeze(sample_result["masks"], 1)
        return sample_result
    
    def squeeze_dimention_for_batch_result(batch_result):
        return [__class__.squeeze_dimention_for_batch_result(x) for x in batch_result]

    # tensor numpy
    @staticmethod
    def tensor_to_numpy(data):
        return data.to("cpu").detach().numpy()
    
    @staticmethod
    def tensor_to_numpy_for_sample_image(sample_image):
        return __class__.tensor_to_numpy(sample_image).transpose((1, 2, 0))

    @staticmethod
    def tensor_to_numpy_for_batch_data(batch_data):
        images, targets = batch_data
        images = [__class__.tensor_to_numpy_for_sample_image(image) for image in images]
        targets = [__class__.tensor_to_numpy_for_sample_target(target) for target in targets]
        return (images, targets)

    @staticmethod
    def tensor_to_numpy_for_batch_result(batch_result):
        return [__class__.tensor_to_numpy_for_sample_result(sample_result) for sample_result in batch_result]

    @staticmethod 
    def tensor_to_numpy_for_sample_target(sample_target):
        labels, boxes, masks = __class__.dict_to_tuple_for_target(sample_target)
        labels = __class__.tensor_to_numpy(labels)
        boxes = __class__.tensor_to_numpy(boxes)
        masks = __class__.tensor_to_numpy(masks)
        return __class__.tuple_to_dict_for_target(labels, boxes, masks)
    
    @staticmethod
    def tensor_to_numpy_for_sample_data(sample_data):
        image, target = sample_data
        return (__class__.tensor_to_numpy_for_sample_image(image), __class__.tensor_to_numpy_for_sample_target(target))
    
    @staticmethod
    def tensor_to_numpy_for_sample_result(sample_result):
        scores, labels, boxes, masks = __class__.dict_to_tuple_for_result(sample_result)
        scores = __class__.tensor_to_numpy(scores)
        labels = __class__.tensor_to_numpy(labels)
        boxes = __class__.tensor_to_numpy(boxes)
        masks = __class__.tensor_to_numpy(masks)
        return __class__.tuple_to_dict_for_result(scores, labels, boxes, masks)
    
    # dict - tuple
    @staticmethod
    def dict_to_tuple_for_target(sample_target):
        return sample_target["labels"], sample_target["boxes"], sample_target["masks"]
    
    @staticmethod
    def dict_to_tuple_for_result(sample_result):
        return  sample_result["scores"], sample_result["labels"], sample_result["boxes"], sample_result["masks"]

    @staticmethod
    def tuple_to_dict_for_result(scores, labels, boxes, masks):
        return {"scores":scores, "labels":labels, "boxes":boxes, "masks":masks}
    
    @staticmethod
    def tuple_to_dict_for_target(labels, boxes, masks):
        return {"labels":labels, "boxes":boxes, "masks":masks}

class Evaluator:
    
    @staticmethod
    def compare_result_and_gt(result, sample, class_num, iou_threshold):
        """
            [Operation]
                iou_threshold를 기준으로 각각의 모델의 객체 검출 결과와 정답을 대응 정보를 반환한다.

            [Args]
                * result: (__Dict[str, Any]__): MaskRCNN의 단일 출력 데이터
                    {
                            "scores": (__Numpy(M, dtype = float32)__),          # M = detected instance num (candidate num)
                            "labels": (__Numpy(M, dtype = uint8)__),
                            "boxes": (__Numpy(M, 4, dtype = float32)__),
                            "masks": (__Numpy(M, H, W, dtype = float32)__),
                    }
                * sample: (__Tuple[Image, Target]__): MaskRCNN의 단일 입력 데이터
                    - Image: (__Numpy(N, 3, H, W)__)                            # M = target instance num
                    - Target: (__Dict[str, Any]__)
                        {
                            "labels": (__Numpy(N, dtype = uint8)__),
                            "boxes": (__Numpy(N, 4, dtype = float32)__),
                            "masks": (__Numpy(N, H, W, dtype = float32)__),
                        } 


            [Result]
                * result: (__Tuple[Candidates, Targetsm, Iou_List]__): score 순으로 정렬된 candidates
                    - Candidates: (__Dict[str, Any]__)
                        {
                            "scores": (__Numpy(M, dtype = float32)__),          # M = detected instance num (candidate num)
                            "labels": (__Numpy(M, dtype = uint8)__),
                            "boxes": (__Numpy(M, 4, dtype = float32)__),
                            "masks": (__Numpy(M, H, W, dtype = float32)__),
                        }
                    - Targets:  (__Dict[str, Any]__) : candidate에 대응되는 target
                        {
                            "labels": (__Numpy(M, dtype = uint8)__),
                            "boxes": (__Numpy(M, 4, dtype = float32)__),
                            "masks": (__Numpy(M, H, W, dtype = float32)__),
                        }
                    - Iou_List:  (Numpy(N, dtype = float32)__) : candidate와 대응되는 target 간의 iou 리스트
        """
        image, target = sample
        
        # 탐색 결과와 정답 결과를 쌍 짓기 (누가 무엇을 탐지하는지)
        pairs = MH.pair_up_instances(result, target, class_num, iou_threshold)
        c_indexes = [pair[0] for pair in pairs] # detection 결과 idx 리스트
        t_indexes = [pair[1] for pair in pairs] # 대응되는 target(gt)의 idx 리스트 (없으면 -1)

        c_scores, c_labels, c_boxes, c_masks = Converter.dict_to_tuple_for_result(result)
        t_labels, t_boxes, t_masks = Converter.dict_to_tuple_for_target(target)

        # target = -1 인 경우에 대한 값 추가
        t_labels = np.append(t_labels, [0], axis = 0) # 배경 추가
        t_boxes = np.append(t_boxes, [[-1, -1, -1, -1]], axis = 0)
        t_masks = np.append(t_masks, [np.zeros(t_masks[0].shape)], axis = 0)

        # candidate 정보를 candidate indexes에 맞게 순서 정리
        c_scores, c_labels, c_boxes, c_masks = c_scores[c_indexes], c_labels[c_indexes], c_boxes[c_indexes], c_masks[c_indexes]

        # gt 정보를 candidate의 대응되는 idx에 따라 정리
        t_labels, t_boxes, t_masks = t_labels[t_indexes], t_boxes[t_indexes], t_masks[t_indexes]

        candidates = {
            "scores" : c_scores,
            "labels" : c_labels,
            "boxes" : c_boxes,
            "masks" : c_masks,
        }

        targets = {
            "labels" : t_labels,
            "boxes" : t_boxes,
            "masks" : t_masks
        }

        iou_list = np.array([MH.iou(c, t) for c, t in zip(c_masks, t_masks)])

        return (candidates, targets, iou_list)

class Visualizer:
    @staticmethod
    def show_comparison_result(candidates, targets, iou_list):
        """
            검출 결과 (candidate)와 그에 대응되는 정답 객체(target)를 나란히 출력한다.
        """
        c_scores, c_labels, c_boxes, c_masks = candidates["scores"], candidates["labels"], candidates["boxes"], candidates["masks"]
        t_labels, t_boxes, t_masks = targets["labels"], targets["boxes"], targets["masks"]
        
        n = len(c_scores)
        fig = plt.figure(figsize = (6, 3*n))


        for i, (c_score, c_label, c_box, c_mask, t_label, t_box, t_mask, iou) in enumerate(
            zip(c_scores, c_labels, c_boxes, c_masks, t_labels, t_boxes, t_masks, iou_list)
        ):
            # prediction instance
            sub = fig.add_subplot(n, 2, 2*i+1)
            sub.imshow(c_mask)
            sub.set_title("Prediction(%s, %.3f, %.3f)"%(VOC_LABEL[c_label], c_score, iou), fontdict={"fontsize":8})

            # Corresponding instance
            sub = fig.add_subplot(n, 2, 2*i+2)
            sub.imshow(t_mask)
            sub.set_title(f"GT({VOC_LABEL[t_label]})", fontdict={"fontsize":8})

class SimpleTorchTask(TorchTask):

    def get_simple_dataset(self, batch_size = 1):
        t_dataset = self.get_train_dataset()
        t_dataloader = self.get_train_dataloader(t_dataset, batch_size = batch_size)
        v_dataset = self.get_val_dataset()
        v_dataloader = self.get_val_dataloader(v_dataset, batch_size = batch_size)
        return t_dataset, t_dataloader, v_dataset, v_dataloader

class MaskRCNN(SimpleTorchTask):
    CLASS_NUM = 21
    SCORE_THRESHOLDS = [0.1*i for i in range(10, 0, -1)]
    MASK_THRESHOLD = 0.5
    
    TRIM_SCORE_THRESHOLD = 0.8 # 모델의 추론 결과물에 대한 score 임계치
    TRIM_MASK_THRESHOLD = 0.5 # 모델의 mask 결과물을 바이너리로 변환할 때 픽셀 별 임계치
    NMS_IOU_THRESHOLD = 0.3 # Non Maximum Supression 수행 시 iou 임계치

    MAP_IOU_THRESHOLD = 0.1

    def __init__(self, name, roi_pooling_size, pre_trained_model = ""):
        super().__init__(name)
        self.roi_pooling_size = roi_pooling_size
        self.pre_trained_model = pre_trained_model # root dir 내에 pretrined model file name

    # Element
    @overrides
    def get_model(self):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
        model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features=1024, out_features=21, bias=True)
        model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features=1024, out_features=84, bias=True)
        model.roi_heads.mask_predictor.mask_fcn_logits = torch.nn.Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))
        for param in model.backbone.body.layer1.parameters():
            param.requires_grad = False
        for param in model.backbone.body.layer2.parameters():
            param.requires_grad = False
        for param in model.backbone.body.layer3.parameters():
            param.requires_grad = False
        if self.pre_trained_model:
            path = self.get_root_dir()/self.pre_trained_model
            model.load_state_dict(torch.load(path))

        model.roi_heads.mask_roi_pool.output_size = self.roi_pooling_size
        return model

    @overrides
    def get_optimizer(self, parameters):
        return torch.optim.SGD(parameters, lr=0.001, momentum=0.9, weight_decay=0.0005)

    @overrides
    def get_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)

    # dataset
    @overrides
    def get_train_dataset(self):
        return Dataset(root = "/home/jovyan/data/VOCdevkit/VOC2012", image_set = "train", 
            cropping = False, mask_expanding = False, flip = False, jitter = False, expanding_size = 0)
    @overrides
    def get_val_dataset(self):
        return Dataset(root = "/home/jovyan/data/VOCdevkit/VOC2012", image_set = "val", 
            cropping = False, mask_expanding = False, flip = False, jitter = False, expanding_size = 0)
        
    # dataloader
    @staticmethod
    def GET_DATALOADER(dataset, batch_size = 1, world_size = 1, rank = 0, num_workers = 1, shuffle = True):
        def collate_batch(batch):
            images = [sample[0] for sample in batch]
            targets = [sample[1] for sample in batch]
            
            return images, targets
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas = world_size, rank = rank, shuffle = shuffle, seed = 0, drop_last = False)
        return torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, collate_fn=collate_batch, sampler = sampler)

    @overrides
    def get_train_dataloader(self, batch_size = 1, world_size=1, rank=0):
        dataset = self.get_train_dataset()
        return __class__.GET_DATALOADER(dataset, batch_size = batch_size, world_size = 1, rank = 0, num_workers = 10, shuffle = True)

    @overrides
    def get_val_dataloader(self, batch_size = 1, world_size=1, rank=0):
        dataset = self.get_val_dataset()
        return __class__.GET_DATALOADER(dataset, batch_size = batch_size, world_size = 1, rank = 0, num_workers = 10, shuffle = True)

    def get_dataset(self, type = "train"):
        if type == "train":
            return self.get_train_dataset()
        elif type == "val":
            return self.get_val_dataset()

    def get_dataloader(self, type = "train"):
        if type == "train":
            return self.get_train_dataloader()
        elif type == "val":
            return self.get_val_dataloader()

    # Device
    @overrides
    def to_for_sample_data(self, sample_data, device):
        image, target = sample_data
        image = image.detach().to(device)
        target = {k:v.detach().to(device) for k, v in target.items()}
        return (image, target)
    
    @overrides
    def to_for_batch_data(self, batch_data, device):
        images, targets = batch_data
        images = [image.detach().to(device) for image in images]
        targets = [{k:v.detach().to(device) for k, v in t.items()} for t in targets]
        return (images, targets)

    @overrides
    def to_for_sample_result(self, sample_result, device):
        return {k:v.detach().to(device) for k, v in sample_result.items()}

    @overrides
    def to_for_batch_result(self, batch_result, device):
        return [__class__.to_for_sample_result(x, device) for x in batch_result]

    # Context
    @overrides
    def before_loss_eval(self, model):
        model.train()

    @overrides
    def feed_for_batch_data(self, model, batch_data):
        images, targets = batch_data
        return model(images, targets)

    def feed_for_sample_data(self, model, sample_data):
        image, target = sample_data
        batch_data = ([image], [target])
        result = self.feed_for_batch_data(model, batch_data)
        
        if isinstance(result, list):
            return result[0] # eval
        else:
            return result # train

    @overrides
    def result_to_loss(self, result):
        v = torch.stack([value for value in result.values()])
        return v @ v ** (1/2)
        # return torch.stack([value for value in result.values()]).sum()

    def is_main(self, world_size, rank, main_rank = 0):
        return (world_size == 1) or (1 < world_size and rank == main_rank)

    def trim_for_batch_result(self, batch_result, score_threshold = TRIM_SCORE_THRESHOLD, mask_threshold = TRIM_MASK_THRESHOLD, nms_iou_threshold = NMS_IOU_THRESHOLD):
        return [self.trim_for_sample_result(sample_result) for sample_result in batch_result]

    def trim_for_sample_result(self, sample_result, score_threshold = TRIM_SCORE_THRESHOLD, mask_threshold = TRIM_MASK_THRESHOLD, nms_iou_threshold = NMS_IOU_THRESHOLD):
        """
        [Args]
            * sample_result: MaskRCNN의 output 형태 
            * score_threshold: (__float__)
            * mask_threshold: (__float__)
            * nms_iou_threshold: (__float__)
        [Operation]
            * 추론 결과에 스코어 필터링과 NMS 등을 적용
        """
        
        scores, labels, boxes, masks = Converter.dict_to_tuple_for_result(sample_result)
        
        # binary mask
        masks = mask_threshold < sample_result["masks"]
        
        # score filtering
        survive = score_threshold < scores 
        scores, labels, boxes, masks = scores[survive], labels[survive], boxes[survive], masks[survive]
        
        # NMS    
        survive = MH.get_nms_index(masks, scores, iou_threshold = nms_iou_threshold)
        scores, labels, boxes, masks = scores[survive], labels[survive], boxes[survive], masks[survive]
        
        return Converter.tuple_to_dict_for_result(scores, labels, boxes, masks)

    # Evaluation
    def map(self,  type = "val", iou_threshold = 0.5, device = "cpu", batch_size = 1):

        model = self.get_model().eval()
        model.to(device)

        if type == "train":
            dataloader = self.get_val_dataloader(batch_size = batch_size)
        elif type == "val":
            dataloader = self.get_train_dataloader(batch_size = batch_size)

        counts = []
        for batch_data in tqdm(dataloader):
            batch_data = self.to_for_batch_data(batch_data, device)
            results = self.feed_for_batch_data(model, batch_data)
            results = self.trim_for_batch_result(results)

            # 클래스 별로 GT, TP, Detection 개수 카운트하여 counts에 추가
            for result, target in zip(results, batch_data[1]):
                result = Converter.tensor_to_numpy_for_sample_result(result)
                target = Converter.tensor_to_numpy_for_sample_target(target)

                p_scores, p_labels, p_boxes, p_masks = Converter.dict_to_tuple_for_result(result)
                gt_labels, gt_boxes, gt_masks = Converter.dict_to_tuple_for_target(target)

                count_result = MH.count_result(p_masks, p_labels, p_scores, gt_masks, gt_labels, 
                                            class_num = __class__.CLASS_NUM, 
                                            score_thresholds = __class__.SCORE_THRESHOLDS,
                                            iou_threshold = iou_threshold)
                counts.append(count_result)
        
        # 카운트 결과 종합
        detected_num_list = [count["detected_num"] for count in counts]
        gt_num_list = [count["gt_num"] for count in counts]
        tp_num_list = [count["tp_num"] for count in counts]

        detected_num = np.stack(detected_num_list).sum(0)
        gt_num = np.stack(gt_num_list).sum(0)
        tp_num = np.stack(tp_num_list).sum(0)

        # MAP 계산 후 반환
        return MH.ap_per_class(detected_num, gt_num, tp_num).mean()

    def show_sample_data(self, idx, type = "val"):
        sample_data = self.get_dataset(type)[idx] # idx 번째 샘플 데이터
        image, target = Converter.tensor_to_numpy_for_sample_data(sample_data)
        aggregated_gt_masks = self.aggregate_mask(target["masks"].squeeze())
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.gca().set_title("Image", fontdict={"fontsize":8})
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)

        plt.subplot(1, 2, 2)
        plt.imshow(aggregated_gt_masks)
        plt.gca().set_title("Ground Truth", fontdict={"fontsize":8})
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)


    def show_sample_result(self, idx, type = "val", device = "cpu"):
        
        # model
        model = self.get_model().eval().to(device)

        # data
        sample_data = self.to_for_sample_data(self.get_dataset(type)[idx], device) # idx 번째 샘플 데이터(device 적용)
        
        # result
        
        sample_result = self.feed_for_sample_data(model, sample_data)
        sample_result = self.trim_for_sample_result(sample_result)

        if len(sample_result["labels"]) == 0:
            return None

        image, target = Converter.tensor_to_numpy_for_sample_data(sample_data)
        aggregated_gt_masks = self.aggregate_mask(target["masks"].squeeze())

        predicted_masks = Converter.tensor_to_numpy_for_sample_result(sample_result)["masks"].squeeze()
        aggregated_predicted_masks = self.aggregate_mask(predicted_masks)
        
        n = len(predicted_masks)
        plt.subplot((n+2)//3+1, 3, 1)
        plt.imshow(image)
        plt.gca().set_title("Image", fontdict={"fontsize":8})
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)

        plt.subplot((n+2)//3+1, 3, 2)
        plt.imshow(aggregated_gt_masks)
        plt.gca().set_title("Ground Truth", fontdict={"fontsize":8})
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)

        plt.subplot((n+2)//3+1, 3, 3)
        plt.imshow(aggregated_predicted_masks)
        plt.gca().set_title("All Predictions", fontdict={"fontsize":8})
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)

        for i, predicted_mask in enumerate(predicted_masks):
            plt.subplot((n+2)//3+1, 3, 4+i)
            plt.imshow(predicted_mask)
            plt.gca().set_title(f"Prediction({i+1})", fontdict={"fontsize":8})
            plt.gca().axes.xaxis.set_visible(False)
            plt.gca().axes.yaxis.set_visible(False)


    def show_sample_result_one_instance(self, idx, instance_idx, type = "val", device = "cpu", iou_threshold = None):
        if iou_threshold == None:
            iou_threshold = __class__.MAP_IOU_THRESHOLD
        
        # model
        model = self.get_model().eval().to(device)

        # data
        sample_data = self.to_for_sample_data(self.get_dataset(type)[idx], device) # idx 번째 샘플 데이터(device 적용)
        
        # result    
        sample_result = self.feed_for_sample_data(model, sample_data)
        sample_result = self.trim_for_sample_result(sample_result)

        if len(sample_result["labels"]) == 0:
            return None

        # to numpy
        sample_result = Converter.tensor_to_numpy_for_sample_result(sample_result)
        image, target = Converter.tensor_to_numpy_for_sample_data(sample_data)

        # 대응되는 gt instance idx
        prediction_gt_pair = MH.pair_up_instances(sample_result, target, __class__.CLASS_NUM, iou_threshold = iou_threshold)
        gt_idx = prediction_gt_pair[instance_idx][1]

        
        aggregated_gt_masks = self.aggregate_mask(target["masks"].squeeze()[gt_idx:gt_idx+1])

        predicted_masks = sample_result["masks"].squeeze()
        aggregated_predicted_masks = self.aggregate_mask(predicted_masks[instance_idx:instance_idx+1])
        
        print("Label: ", sample_result["labels"][instance_idx])
        print("Score: ", sample_result["scores"][instance_idx])
        print("IoU  : ", MH.iou(predicted_masks[instance_idx], target["masks"].squeeze()[gt_idx]))


        n = len(predicted_masks)
        plt.subplot((n+2)//3+1, 3, 1)
        plt.imshow(image)
        plt.gca().set_title("Image", fontdict={"fontsize":8})
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)

        plt.subplot((n+2)//3+1, 3, 2)
        plt.imshow(aggregated_gt_masks)
        plt.gca().set_title("Ground Truth", fontdict={"fontsize":8})
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)

        plt.subplot((n+2)//3+1, 3, 3)
        plt.imshow(aggregated_predicted_masks)
        plt.gca().set_title(f"Prediction({instance_idx+1})", fontdict={"fontsize":8})
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        
    def aggregate_mask(self, masks):
        """
        [Operation]
            mask와 color를 일대일 매핑하여 하나의 이미지로 통합하여 반환
        """
        
        # mask와 동일한 size의 빈 이미지 생성
        image_shape = (masks.shape[1], masks.shape[2], 3)
        white_canvas = np.ones(image_shape, dtype = np.float32)
        
        # 마스크를 모두 통합
        masked_canvas = MH.aggregate(white_canvas, masks, alpha = 1)
        return masked_canvas
        

    def get_loss_dict(self):
        path = self.get_root_dir()/"loss_dict.p"
        with open(path, "rb") as file:
            loss_dict = pickle.load(file)
        return loss_dict

class MaskRCNN_Mask_Expanding(MaskRCNN):     
    @overrides
    def __init__(self, name, expanding_size, pre_trained_model = ""):
        super().__init__(name, pre_trained_model)
        self.expanding_size = expanding_size

    # dataset & dataloader
    @overrides
    def get_train_dataset(self):
        return Dataset(root = "/home/jovyan/data/VOCdevkit/VOC2012", image_set = "train", 
            cropping = False, mask_expanding = True, flip = False, jitter = False, expanding_size = self.expanding_size)


class MaskRCNN_Boundary_Reinforcement(MaskRCNN):
    def __init__(self, name, roi_pooling_size, boundary_size, pre_trained_model = ""):
        super().__init__(name, roi_pooling_size, pre_trained_model)
        self.boundary_size = boundary_size

        def init_train_mode(self, model):
            model.train()
    
    @overrides
    def before_train_one_epoch(self, model):
        # 매 train 전에 torch 코드의 boundary loss 부분 수정
        BOUNDARY_LOSS = True
        BOUNDARY_SIZE = self.boundary_size
        model.train()

    @overrides
    def after_train_one_epoch(self, model):
        # boudary loss 제거
        BOUNDARY_LOSS = False

class MaskRCNN_Flip(MaskRCNN):     
    @overrides
    def get_train_dataset(self):
        return Dataset(root = "/home/jovyan/data/VOCdevkit/VOC2012", image_set = "train", 
            cropping = False, mask_expanding = False, flip = True, jitter = False, expanding_size = 0)

class MaskRCNN_Jitter(MaskRCNN):     
    @overrides
    def get_train_dataset(self):
        return Dataset(root = "/home/jovyan/data/VOCdevkit/VOC2012", image_set = "train", 
            cropping = False, mask_expanding = False, flip = False, jitter = True, expanding_size = 0)

class MaskRCNN_Cropping(MaskRCNN):     
    @overrides
    def get_train_dataset(self):
        return Dataset(root = "/home/jovyan/data/VOCdevkit/VOC2012", image_set = "train", 
            cropping = True, mask_expanding = False, flip = False, jitter = False, expanding_size = 0)

class MaskRCNN_C_F_J(MaskRCNN):     
    @overrides
    def get_train_dataset(self):
        return Dataset(root = "/home/jovyan/data/VOCdevkit/VOC2012", image_set = "train", 
            cropping = True, mask_expanding = False, flip = True, jitter = True, expanding_size = 0)