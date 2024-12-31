# Unsupervised Segmentation
# Source: https://github.com/Yonv1943/Unsupervised-Segmentation/tree/master

import os
import time
from collections import Counter
import argparse

import cv2
import numpy as np
from skimage import segmentation
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

class SelfencodingNet(nn.Module):
    """
    Custom neural network for unsupervised segmentation.
    """
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(SelfencodingNet, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
        )

    def forward(self, x):
        return self.seq(x)


def parse_args():
    """
    Parse command-line arguments.
    """
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, '..')  # 假设 Nano 是当前文件夹的上级目录
    default_img_dir = os.path.join(root_dir, 'SEM dataset/Dataset for deeplearning/images')
    output_img_dir = os.path.join(root_dir, 'Output/un_pred')

    parser = argparse.ArgumentParser(description="Unsupervised Image Segmentation")
    parser.add_argument('--img_dir', type=str, default=default_img_dir,
                        help='Directory containing input images.')
    parser.add_argument('--out_dir', type=str, default=output_img_dir, help='Directory to save output images.')
    parser.add_argument('--train_epoch', type=int, default=64, help='Number of training epochs.')
    parser.add_argument('--mod_dim1', type=int, default=64, help='Number of filters for the first layer.')
    parser.add_argument('--mod_dim2', type=int, default=32, help='Number of filters for the second layer.')
    parser.add_argument('--min_label_num', type=int, default=4, help='Minimum number of labels to stop training.')
    parser.add_argument('--max_label_num', type=int, default=256, help='Maximum number of labels to display results.')
    return parser.parse_args()


def run(image_filename, args):
    """
    Main function to run the segmentation on a single image.
    """
    start_time0 = time.time()
    torch.manual_seed(1943)
    np.random.seed(1943)

    # Load image
    image = cv2.imread(image_filename)

    # Initial segmentation using Felzenszwalb
    seg_map = segmentation.felzenszwalb(image, scale=32, sigma=0.5, min_size=64)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0] for u_label in np.unique(seg_map)]

    # Prepare input tensor
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    tensor = image.transpose((2, 0, 1)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(tensor[np.newaxis, :, :, :]).to(device)

    # Initialize model, loss, and optimizer
    model = SelfencodingNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)

    # Flatten image for visualization
    image_flatten = image.reshape((-1, 3))
    color_avg = np.random.randint(255, size=(args.max_label_num, 3))
    show = image

    # Training loop
    start_time1 = time.time()
    model.train()
    for batch_idx in tqdm(range(args.train_epoch)):
        optimizer.zero_grad()
        output = model(tensor)[0]
        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
        target = torch.argmax(output, 1)
        im_target = target.data.cpu().numpy()

        # Refine labels
        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]

        # Backpropagation
        target = torch.from_numpy(im_target).to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update visualization
        un_label, lab_inverse = np.unique(im_target, return_inverse=True)
        if un_label.shape[0] < args.max_label_num:
            img_flatten = image_flatten.copy()
            if len(color_avg) != un_label.shape[0]:
                color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int64) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                img_flatten[lab_inverse == lab_id] = color
            show = img_flatten.reshape(image.shape)

        if len(un_label) < args.min_label_num:
            break

    # Save results
    time0 = time.time() - start_time0
    time1 = time.time() - start_time1
    print(f'PyTorchInit: {time0:.2f}\nTimeUsed: {time1:.2f}')
    show = show[:, :, 0]
    pixels = list(Counter(show.flatten()))

    show[show <= np.mean(pixels)] = 0
    show[show > np.mean(pixels)] = 255
    output_path = os.path.join(args.out_dir, f"pred_{os.path.basename(image_filename).split('.')[0]}.png")
    plt.imsave(output_path, show, cmap="gray")

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for image_filename in sorted(os.listdir(args.img_dir)):
        image_path = os.path.join(args.img_dir, image_filename)
        if os.path.isfile(image_path):
            print(f'Processing: {image_path}')
            run(image_path, args)
