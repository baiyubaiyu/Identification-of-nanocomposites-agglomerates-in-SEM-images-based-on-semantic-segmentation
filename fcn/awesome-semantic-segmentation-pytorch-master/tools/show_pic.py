'''
    用来显示图片和统计图像像素值分布
'''

import matplotlib.pyplot as plt
from tools.read_image import pixel_hist
import imageio

image = imageio.imread('/root/CV_Project/awesome-semantic-segmentation-pytorch-master/unsupervised/image/pred.png')


plt.imshow(image)
plt.show()
pixel_hist(image)
# plt.imsave('/home/baiyu/Projects/Sem_image_segmentation/catt.png', image)
#
# image = plt.imread('/home/baiyu/Projects/Sem_image_segmentation/catt.png') * 255
# print(image.unique())
# print(np.unique(image))