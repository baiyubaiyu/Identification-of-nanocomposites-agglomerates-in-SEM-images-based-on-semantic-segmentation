'''
    可视化分割结果，即将mask和image结合显示出可视图
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize(imgfile,maskfile):
    img = cv2.imread(imgfile)
    mask = cv2.imread(maskfile)
    binary,contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    # print(img.shape)
    img = img[:, :, ::-1]
    img[..., 2] = np.where(mask == 1, 255, img[..., 2])
    return img

img_dir = '/home/baiyu/Data/Test_SEM/pred/image'
mask_dir = '/home/baiyu/Data/Test_SEM/pred/mask'
output_dir = '/home/baiyu/Data/Test_SEM/pred/label'

imgfiles = [os.path.join(img_dir, img_name) for img_name in sorted(os.listdir(img_dir))]
maskfiles = [os.path.join(mask_dir, mask_name) for mask_name in sorted(os.listdir(mask_dir))]

assert len(imgfiles)==len(maskfiles), '"The number of images must be equal to masks"'

for i in range(len(imgfiles)):
    label = visualize(imgfiles[i],maskfiles[i])
    label_name = imgfiles[i].split('/')[-1].split('.')[0]
    plt.imsave(os.path.join(output_dir,label_name) + '.png', label)
