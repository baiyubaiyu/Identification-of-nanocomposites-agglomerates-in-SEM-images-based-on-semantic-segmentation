'''
    计算iou
'''
import torch
import numpy as np

SMOOTH = 1e-6

def iou_numpy(outputs: np.array, labels: np.array):
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou

if __name__ == '__main__':
    a = np.array([[1,1,1,0],
                  [1,1,1,0],
                  [1,1,1,0]],dtype=np.int64)
    b = np.array([[0,1,1,0],
                  [0,1,1,0],
                  [0,1,1,0]],dtype=np.int64)
    print(iou_numpy(b,a))


# def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
#     # You can comment out this line if you are passing tensors of equal shape
#     # But if you are passing output from UNet or something it will most probably
#     # be with the BATCH x 1 x H x W shape
#     # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
#
#     intersection = (outputs & labels).float().sum()  # Will be zero if Truth=0 or Prediction=0
#     union = (outputs | labels).float().sum()  # Will be zzero if both are 0
#
#     iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
#
#     thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
#
#     return thresholded  # Or thresholded.mean() if you are interested in average across the batch