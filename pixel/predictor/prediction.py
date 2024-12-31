'''
    预测程序入口
'''
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
from dataset.image_dataset import ImageDataset
from dataset.patch_dataset_memory import Patch
from dataset.sampleall import sample_all
from metrics.Iou import iou_numpy
from config import param
import cv2
import os
from tqdm import tqdm
from utils.logging import setup_logger
from dataset.read_image import readImage
from math import floor
from dataset.transform.padding import reflect_pad2d

logger = setup_logger('pred_log', './')

''' *************************测试图片读取、以及参数设定************************* '''
DEVICE = param.PREDICT_DEVICE
CONTEXT_AREA = param.PREDICT_CONTEXT_AREA
BATCH_SIZE = param.PREDICT_BATCH_SIZE
OUT_DIR = param.PREDICT_DIR
img_dir = '/root/Data/new_images'
pred_dataset = ImageDataset(img_dir, img_dir, CONTEXT_AREA)

''' *************************加载模型************************* '''
model_name = param.PREDICT_MODEL_NAME
model = torch.load(model_name)

''' *************************测试函数************************* '''
def pred(model, device, test_loader, mask_shape):
    model.eval()
    test_loss = 0
    correct = 0
    y_pred = []
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data = data.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            a = torch.squeeze(pred)
            y_pred += torch.squeeze(pred).cpu().numpy().tolist()

    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(mask_shape)

    return y_pred

''' 遍历测试集，进行预测 '''
logger.info(''' Prediction ''')
logger.info('Num:', len(pred_dataset))
for i in range(len(pred_dataset)):
    img_name = os.path.splitext(os.path.basename(pred_dataset.item_filenames[i]['img']))[0]
    logger.info('pic{}:{}'.format(i, img_name))
    image, mask = pred_dataset[i]
    mask_shape = mask.shape[1:]
    patch = sample_all(image/255, mask, CONTEXT_AREA, pred=True)
    ''' 构建每一张图的dataset '''
    pred_patch_dataset = Patch(patch, patch)
    pred_loader = data.DataLoader(pred_patch_dataset, BATCH_SIZE)
    ''' 测试 '''
    y_pred = pred(model, DEVICE, pred_loader, mask_shape=mask_shape)

    print(Counter(y_pred.flatten()))
    # y_pred[y_pred > 0] = 255


    # 保存图片到该文件夹，并命名为pred_原名
    cv2.imwrite(os.path.join(OUT_DIR,'pred_{}.png').format(img_name), y_pred)
    # cv2.imwrite('/root/Data/SEM/pixel/test/pred/pred_{}.png'.format(img_name), y_pred)



