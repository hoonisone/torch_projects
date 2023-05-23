import torch
import torchvision
import numpy as np
from typing import Dict, List, Any
import cv2

def random_cropping(image1, image2, size):
    """
        [Args]
            image (torch.tensor[N, H, W]): 
            mask (torch.tensor[M, H, W]): each pixel has a label

        [Do]
            두 tensor 이미지를 동일한 영역으로 random coppring 처리하여 반환한다.

        [Return]
            cropped_image (torch.tensor[N, H', W']): 
            cropped_mask (torch.tensor[M, H', W']): each pixel has a label
    """

    # 두 이미지를 합친다.
    data = torch.vstack([image1, image2])

    # 원하는 사이즈로 random cropping 한다.
    data = torchvision.transforms.RandomCrop(size = size)(data)

    # 다시 원래 차원대로 구분하여 반환한다.
    n = image1.shape[0]
    return data[:n], data[n:]

def extract_box_from_binary_mask(binary_mask):
    """
        [Args]
            Tensor[bool](h, w) binary_mask : binary tensor 
        
        [Do]
            값이 있는 영역에 대해 바운딩 박스를 좌표로 반환한다.

        [Return]
            torch.tensor[int] bounding_box: [x1, y1, x2, y2]
    """
    h, w = binary_mask.shape
    if binary_mask.any() == False:
        
        return torch.tensor([0, 0, w-1, h-1], dtype = torch.float32)
    y_where, x_where = torch.where(binary_mask.type(torch.bool) == True)
    y1, y2 = y_where.min(), y_where.max()
    x1, x2 = x_where.min(), x_where.max()
    
    if y1 == y2:
        y1 = torch.tensor([0, y1-1]).max()
        y2 = torch.tensor([h-1, y2+1]).min()

    if x1 == x2:
        x1 = torch.tensor([0, x1-1]).max()
        x2 = torch.tensor([w-1, x2+1]).min()

    # return torch.tensor([0, 0, w-1, h-1], dtype = torch.float32)
    return torch.stack([x1, y1, x2, y2]).type(torch.float32)

def iou(mask1, mask2):
    """
    [Operation]
	    * 두 마스크에 대해 Intersection over Union값을 계산하여 반환한다.

    [Args]
        * mask1: (__Numpy(H, W)__):

        * mask2: (__Numpy(H, W)__):
        
    [Algorithm]

    [Result]
        * result: (__float__)
        
    """
    intersection = (mask1 * mask2).sum()
    union = (mask1 + mask2).sum() - intersection
    return intersection/union

def pair_up_instances(candidates, targets, class_num, iou_threshold = 0.5):
        """
        [Operation]
            candidates 각각에 대해 targets에서 어떤것을 검출하려 했는지를 판단한다.
            - 그 결과를 index 쌍으로 반환한다.
            - 대상이 없는 경우 -1로 반환한다.

        [Args]
            * candidates: (__Dict[str, any]__): detected instances
                {
                    "scores": Numpy(N, dtype = float32)             # N = detected instance num
                    "labels": Numpy(N, 4, dtype = float32)               
                    "masks": Numpy(N, H, W, dtype = uint8)          # H, W = height, width of detected mask
                }

            * targets: (__Dict[str, any]__): target instances
                {
                    "labels": Numpy(M, 4, dtype = float32),         # M = ground_truth instance num
                    "masks": Numpy(M, H, W, dtype = uint8)
                }

            * class_num: (__int__) : the number of total classes
                101

            * iou_threshold: (__float32__): if iou > this, then detection is considered as positive
                0.5

        [Algorithm]
            1) 후보 리스트에서 score가 가장 높은 후보를 뽑는다.
            2) 대상 리스트에서 후보와 label이 같고 iou가 가장 큰 target을 찾는다.
            3) target이 존재하고 iou가 임계치 보다 같거나 크면 후보와 대상을 연결하고 각 리스트에서 제거한다.
                그렇지 않은 경우 후보와 (-1)을 쌍을 짓고 후보 리스트에서 후보를 제거한다. 
            4) 후보 리스트가 빌 때 까지 반복한다. 

        [Return]
        * return: (__Numpy(N, 2, dtype=uint8)__):
            numpy.ndarray([
                [candidate_idx1, target_idx4],
                [candidate_idx2, target_idx3],
                ... 
            ])
        """
        
        # 1. data 정리 ##########################################################
        # candidates data
        c_scores = candidates["scores"]
        c_labels = candidates["labels"]
        c_masks = candidates["masks"]

        # target data
        t_labels = targets["labels"]
        t_masks = targets["masks"]

        # 2. sort candidates for score #########################################
        # score에 대해 내림차순으로 정렬
        c_indexes = np.argsort(c_scores)[::-1]

        # 3. classify target for classes
        # class별 target index list dict 생성
        class_t_indexes_dict = [[] for i in range(class_num)]
        for idx, label in enumerate(t_labels):
            class_t_indexes_dict[label].append(idx)

        # 4. pair up ###########################################################
        result = []
        
        for c_idx in c_indexes:
            # c_idx에 맞는 label과 mask 선택
            c_label = c_labels[c_idx]
            c_mask = c_masks[c_idx]

            # 레이블이 같은 대상의 인덱스 리스트 구하기
            t_indexes = class_t_indexes_dict[c_label]

            if t_indexes: # 대상이 있는 경우
                iou_list = np.array([iou(c_mask, t_masks[t_idx]) for t_idx in t_indexes])
                max_iou = iou_list.max()
                target_idx = iou_list.argmax()
                if iou_threshold <= max_iou: # 임계치를 넘는 경우
                    result.append([c_idx, t_indexes[target_idx]])
                    t_indexes.pop(target_idx)
                    continue    
        
            # if not ((대상 존재) and (임계치 < iou))
            result.append([c_idx, -1])

        return np.array(result)

def to_binary_by_threshold(data, threshold):
    """
    Numpy mask: binary 데이터로 바뀔 대상 
    threshold: binary 기준 (threshold보다 크면 1 아니면 0)
    """
    return (data > threshold).astype(bool)


def move_up(image, distance):
    if distance == 0:
        return image
    return np.append(image[distance:], np.zeros((distance, image.shape[1])), axis=0)
    
def move_down(image, distance):
    if distance == 0:
        return image
    return np.append(np.zeros((distance, image.shape[1])), image[:-distance], axis=0)
    
def move_left(image, distance):
    return move_up(image.T, distance).T

def move_right(image, distance):
    return move_down(image.T, distance).T

def move(mask, x, y):
    mask = move_right(mask, x) if 0 <= x else move_left(mask, -x)
    mask = move_up(mask, y) if 0 <= y else move_down(mask, -y)
    return mask

def expend_mask(mask, size):
    """
    [Operation]
        mask를 상하좌우로 size pixel만큼 확장한다.
    
    [Args]
        * mask: Numpy(H, W)
    
    [Return]
        * return: Numpy(H, W, dtype = float32)
    """
    expended_mask = np.zeros(mask.shape)
    for y in range(-size, size+1):
        for x in range(-size, size+1):
            move_mask = move(mask, x, y)
            expended_mask += move_mask

    expended_mask = to_binary_by_threshold(expended_mask, 0.5).astype(np.float32)
    return expended_mask

def score_level(score, thresholds):
    f"""
    [Args]
        * score: (__float__): 구간을 확인할 score
        * thresholds: (__Numpy(N, dtype = float32)__): 객체 검출기의 신뢰도에 대한 임계치 리스트 (내림차순)
    [Return]
        * result: (__int__): 몇 번째 임계치 보다 큰지 반환
    """
    for level, threshold in enumerate(thresholds):
        if score >= threshold:
            return level
            
def count_result(p_masks, p_labels, p_scores, gt_masks, gt_labels, class_num, score_thresholds, iou_threshold):
    """
    [Operation]
        객체 검출 결과에 대해 클래스 별 detection num, ground truth num, true positive num 개수를 계산한다.

    [Return]
        * result: (__Dict[str, Any]__):
            {
                "detected_num": (__Numpy(C, S, dtype = int)__): 클래스 및 스코어 구간 별 모델이 검출한 객체 개수 (스코어 = 내림차순)
                "gt_num": (__Numpy(C, dtype = int)__): 클래스 별 ground truth 객체 개수
                "tp_num": (__Numpy(C, S, dtype = int)__): 클래스 및 스코어 구간 별 true positive 개수 (스코어 = 내림차순)
            }

    """
    score_levels = [score_level(score, score_thresholds) for score in p_scores]
    
    #########################################################################
    # prediction과 GT의 idx를 class 별로 구분
    # 클래스 별 detected mask indexes
    detected_indexes_per_c = [[] for i in range(class_num)]
    for idx, label in enumerate(p_labels):
        detected_indexes_per_c[label].append(idx)

    gt_indexes_per_c = [[] for i in range(class_num)]
    for idx, label in enumerate(gt_labels):
        gt_indexes_per_c[label].append(idx)
        
    #########################################################################
    # prediction과 GT를 class별로 카운트
    
    # 클래스 별 탐지한 object 개수
    # detected_num = np.array([len(x) for x in detected_indexes_per_c])

    # 클래스 별 object 개수
    gt_num = np.array([len(x) for x in gt_indexes_per_c])
    

    #########################################################################
    

    tp_num = np.zeros((class_num, len(score_thresholds)))
    detected_num = np.zeros((class_num, len(score_thresholds)))
    for c in range(class_num):
        p_idxes = detected_indexes_per_c[c]
        gt_idxes = gt_indexes_per_c[c]

        if (not p_idxes) or (not gt_idxes):
            continue
        
        for p_idx in p_idxes:
            if not gt_idxes: # 탐색할게 없으면 끝
                break

            iou_list = [iou(p_masks[p_idx], gt_masks[gt_idx]) for gt_idx in gt_idxes]

            max_iou = max(iou_list)
            max_idx = np.array(iou_list).argmax()

            detected_num[c][score_levels[p_idx]] += 1
            if iou_threshold <= max_iou:
                tp_num[c][score_levels[p_idx]] += 1

                gt_idxes.pop(max_idx)
                
    return {"detected_num":detected_num, "gt_num":gt_num, "tp_num":tp_num}


def ap(detected_num, gt_num, tp_num):
    """
    [Operation]
        * 단일 클래스에 대해 모델이 검출한 객체 개수, 전체 객체 개수, 신뢰도 별 positive 개수가 주어졌을 때
        * average precision을 계산하여 반환한다.
    [Args]
        * detected_num: (__Numpy(S)__): score 구간 1, 0.9, ...., 0 따른 모델이 제안한 객체 개수
        * gt_num: (__int__): 전체 데이터 셋에서 단일 클래스에 대한 ground truth 객체 개수
        * tp_num: (__Numpy(S)__): score 구간 1, 0.9, ...., 0 따른 True Positive 개수
    [Return]
        * result: (__Numpy(1)__) average precision
    """

    # score 별 개수 -> score 위로 개수 누적
    for i in range(0, len(tp_num)-1):
        tp_num[i+1] += tp_num[i]
        detected_num[i+1] += detected_num[i]

    # score 구간 별 recall 계산 (오른쪽으로 갈 수록 커진다.)
    recall = np.ones(tp_num.shape) if gt_num == 0 else (tp_num/gt_num)

    # recall 구간 길이 계산
    for i in range(1, len(recall)):
        recall[i] -= recall[i-1]

    precision = np.array([0 if det == 0 else tp/det for det, tp in zip(detected_num, tp_num)])

    # precision을 계단식으로 맞추기
    # recall이 더 높은 영역에서 최대 precision 사용
    for i in range(0, len(precision)):
        precision[i] = max(precision[i:])
    
    # 너비 게산
    return (recall * precision).mean()


def ap_per_class(detected_num, gt_num, tp_num):
    """
    [Operation]
    * 클래스별 ap를 계산하고 평균을 반환
    [Args]
        * detected_num: (__Numpy(C, S, dtype = uint8)__):전체 데이터에 대한 class 및 score level 별 모델의 검출 객체 개수 # C = class num   # S = Score level num
        * gt_num: (__Numpy(C, dtype = uint8)__):전체 데이터에 대한 class 별 ground-truth 객체 개수 
        * tp_num: (__Numpy(C, S, dtype = uint8)__):전체 데이터에 대한 class 및 score level 별 true positive 개수    
            - score level 내림차순 (첫 인덱스가 가장 높은 정확도)
    [Result]
        * result: (__Numpy(C, dtype = float32__): 클래스 별 ap
    """

    ap_list = []
    for c in range(len(detected_num)):
        ap_value = ap(detected_num[c], gt_num[c], tp_num[c])
        ap_list.append(ap_value)

    return np.array(ap_list)


def overlay(image, mask, color, alpha):
    """
    [Operation]
        Combines image and its segmentation mask into a single image.
    [Args]
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,

    [Returns]
        image_combined: (__Numpy(H, W, 3)__) The combined image.

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

colors = [
    [10,  0, 0], [0, 10,  0], [ 0, 0, 10],
    [10, 10, 0], [0, 10, 10], [10, 0, 10],
    [30,  0, 0], [0, 30,  0], [ 0, 0, 30], 
    [30, 30, 0], [0, 30, 30], [30, 0, 30],
    [50,  0, 0], [0, 50,  0], [ 0, 0, 50], 
    [50, 50, 0], [0, 50, 50], [50, 0, 50],
    [70,  0, 0], [0, 70,  0], [ 0, 0, 70], 
    [70, 70, 0], [0, 70, 70], [70, 0, 70],
    [90,  0, 0], [0, 90,  0], [ 0, 0, 90], 
    [90, 90, 0], [0, 90, 90], [90, 0, 90],
]

    


def get_nms_index(masks, scores, iou_threshold: float):
    """
    [Args]
        masks: (__Tensor(N, W, H)__)
        scores: (__Tensor(N, W, H, dtype = float)__)
    [Operation]
        nms 수행 후 살아남는 mask에 대한 idx만 반환
    [Return]
        int list
    """
    candidate = torch.argsort(-scores)
    fix = [] 
    while 0 < len(candidate):
        fixed_idx = candidate[0]
        fix.append(fixed_idx.item())
        survive = []
        for i in candidate[1:]:
            if iou(masks[fixed_idx], masks[i]) < iou_threshold:
                survive.append(i)
        candidate = survive
    return fix

def aggregate(image, masks, alpha = 0.1):
    """
    [Args]
        image = Numpy(3, H, W, dtype = float)
        masks = Numpy(n, H, W, dtype = bool)
        alpha: 투명도
        
    [Operation]
        image위에 마스크를 입힌다.
    """
    for i, mask in enumerate(masks):
        image  = overlay(image, mask, colors[i], alpha)
    
    return image

