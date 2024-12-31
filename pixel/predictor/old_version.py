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
from dataset.read_image import readImage
# from merge.merging import trans_image, merging
# from skimage.segmentation import slic, mark_boundaries
from config import param
import cv2
import os
from tqdm import tqdm

''' 对没有标签的数据 '''
# picture = readImage('/home/baiyu/Dataset/TEST/images/renhuai_6bands_ndvi_ndwi_32bit_10m_test1.tif')
# l = np.zeros(picture.shape[1:])
# cv2.imwrite('/home/baiyu/Dataset/TEST/labels/renhuai_1.png', l)
# ll = readImage('/home/baiyu/Dataset/TEST/labels/renhuai_0.png')
# print(ll.shape)

''' *************************测试图片读取、以及参数设定************************* '''
DEVICE = param.PREDICT_DEVICE
CONTEXT_AREA = param.PREDICT_CONTEXT_AREA
BATCH_SIZE = param.PREDICT_BATCH_SIZE


test_img_dir = '/root/Data/SEM/pixel/test/image'
test_label_dir = '/root/Data/SEM/pixel/test/label'
img_name = test_img_dir + '/' + os.listdir(os.path.join(test_img_dir))[0]  # 这里测试文件里只有一张图
print(img_name)
test_dataset = ImageDataset(test_img_dir, test_label_dir, CONTEXT_AREA)

patches_test = []
pixel_labels_test = []
for i in range(len(test_dataset)):
    image, label = test_dataset[i]
    patch, pixel_label = sample_all(image/255, label, CONTEXT_AREA)
    patches_test = list(patch) # 相当于调用list构造方法
    pixel_labels_test = list(pixel_label) # https://blog.csdn.net/qq_42537915/article/details/103240670
print(len(patches_test), len(pixel_labels_test))

pixel_labels_test = np.array(pixel_labels_test, dtype=int)
# pixel_labels_test[pixel_labels_test == 255] = 1

test_patch_dataset = Patch(patches_test, pixel_labels_test)
test_loader = data.DataLoader(test_patch_dataset, BATCH_SIZE)

''' *************************测试函数************************* '''
def test(model, device, test_loader, mask_shape=(960,1280)):
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
            y_pred += pred.view_as(target).cpu().numpy().tolist()
    ''' loss '''
    test_loss /= len(test_loader.dataset)
    print('Test set loss: {:.6f}'.format(test_loss))
    ''' iou '''
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(mask_shape)
    y_true = np.array(y_true)
    y_true = y_true.reshape(mask_shape)

    print('IOU: {:.2f}%\n'.format(100. * iou_numpy(y_pred, y_true)))
    return y_true, y_pred


''' *************************加载模型及测试************************* '''
model_name = param.PREDICT_MODEL_NAME
model = torch.load(model_name)

y_true, y_pred = test(model, DEVICE, test_loader)
y_pred[y_pred>0] = 1

# 保存图片到该文件夹，并命名为pred_原名
cv2.imwrite('/root/Data/SEM/pixel/test/pred/pred_{}.png'.format(img_name.split('.')[0].split('/')[-1]), y_pred)
#
image = plt.imread(img_name)
# image = trans_image(image)
#
# ''' slic '''
# seg = slic(image, n_segments=param.PREDICT_SLIC, compactness=10, multichannel=True)
# ''' merging '''
# pred_label, merge_seg = merging(y_pred, seg, threshold=param.PREDICT_MERGE)
#
# print('After Merging: \n')
# print('IOU: {:.2f}%\n'.format(100. * iou_numpy(pred_label, y_true)))
#
''' 原图 '''
# mark = mark_boundaries(image, merge_seg)
plt.subplot(221)
plt.imshow(image)
#
''' 标签 '''
plt.subplot(222)
plt.imshow(y_true)
plt.title('label')
#
''' 预测 '''
plt.subplot(223)
plt.imshow(y_pred)
plt.title('pred')
#
# # ''' 预测 '''
# plt.subplot(224)
# plt.imshow(pred_label)
# plt.title('merging')
plt.show()
