'''
    用来显示图片和统计图像像素值分布
'''
import os
import matplotlib.pyplot as plt
b = '/root/Data/new_pred/pixel/pred_EPB05_5000_1.png'
# print(len(os.listdir(b)))
# a = sorted(os.listdir(b))[0]
image = plt.imread(b)
print(image.shape)
# print(Counter(image.flatten()))
plt.imshow(image)
plt.show()
# pixel_hist(image)
# plt.imsave('/home/baiyu/Projects/Sem_image_segmentation/catt.png', image)
#
# img = plt.imread('/home/baiyu/Projects/Sem_image_segmentation/catt.png') * 255
# print(image.unique())
# print(np.unique(image))