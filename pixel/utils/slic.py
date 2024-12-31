'''
    slic超像素分割的效果
'''
from skimage.segmentation import slic,mark_boundaries
import matplotlib.pyplot as plt
image = plt.imread('/home/baiyu/Data/Test_SEM/image/10un_2.jpg')/255
seg = slic(image, n_segments=5000, compactness=10, multichannel=True)
mark = mark_boundaries(image, seg)
plt.imshow(mark)
plt.show()