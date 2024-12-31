'''
    预测程序入口
'''
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

logger = setup_logger('test_log', './')

''' *************************测试图片读取、以及参数设定************************* '''
DEVICE = param.PREDICT_DEVICE
CONTEXT_AREA = param.PREDICT_CONTEXT_AREA
BATCH_SIZE = param.PREDICT_BATCH_SIZE
img_dir = '/root/Data/new_sem/val/images'
mask_dir = '/root/Data/new_sem/val/masks'
test_dataset = ImageDataset(img_dir, mask_dir, CONTEXT_AREA)

''' *************************加载模型************************* '''
model_name = param.PREDICT_MODEL_NAME
model = torch.load(model_name)

''' *************************测试函数************************* '''
def test(model, device, test_loader, mask_shape=(674,1024)):
    model.eval()
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_true += target.cpu().numpy().tolist()
            a = pred.view_as(target)
            y_pred += pred.view_as(target).cpu().numpy().tolist()
    ''' loss '''
    test_loss /= len(test_loader.dataset)
    logger.info('Test set loss: {:.6f}'.format(test_loss))
    ''' iou '''
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(mask_shape)
    y_true = np.array(y_true)
    y_true = y_true.reshape(mask_shape)

    logger.info('IOU: {:.2f}%\n'.format(100. * iou_numpy(y_pred, y_true)))
    return y_true, y_pred

''' 遍历测试集，进行预测 '''
logger.info(''' Test ''')
logger.info('Num:', len(test_dataset))
for i in range(len(test_dataset)):
    img_name = os.path.splitext(os.path.basename(test_dataset.item_filenames[i]['img']))[0]
    logger.info('pic{}:{}'.format(i, img_name))
    image, label = test_dataset[i]
    patch, pixel_label = sample_all(image/255, label, CONTEXT_AREA)
    pixel_label = np.array(pixel_label, dtype=int) # 标签int
    ''' 构建每一张图的dataset '''
    test_patch_dataset = Patch(patch, pixel_label)
    test_loader = data.DataLoader(test_patch_dataset, BATCH_SIZE)
    ''' 测试 '''
    y_true, y_pred = test(model, DEVICE, test_loader)
    y_pred[y_pred > 0] = 255

    # 保存图片到该文件夹，并命名为pred_原名
    cv2.imwrite('/root/Data/new_pred/pixel/pred_{}.png'.format(img_name), y_pred)



