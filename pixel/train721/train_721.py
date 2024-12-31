'''
    训练程序入口
'''
import  sys
sys.path.append('/root/CV_Project/pixel_method')
from dataset.patch_dataset import PatchDataset
import os
import random
import torch
import torch.utils.data as data
from train_test_define.train_test_classification import train, val, test
from sklearn.model_selection import train_test_split
from config import param
from utils.logging import setup_logger


''' *************************训练参数************************* '''
if param.TRAIN721_MODEL_NET == 'dilatedconv':
    from net.dilatedconv import Net
else:
    from net.convnet import Net

DEVICE = param.TRAIN721_DEVICE
EPOCH = param.TRAIN721_EPOCH
BATCH_SIZE = param.TRAIN721_BATCH_SIZE
CONTEXT_AREA = param.TRAIN721_CONTEXT_AREA
LR = param.TRAIN721_LR

model_outpath = param.OUTPUTDIR_MODEL
logger = setup_logger('train_log', param.OUTPUTDIR_LOOGER)

''' 先把所有样本的文件名列出来 '''
tuanju_dir = param.TUANJU_DIR
other_dir = param.OTHER_DIR

tunaju_filename = [os.path.join(tuanju_dir, filename) for filename in sorted(os.listdir(tuanju_dir))]
other_filenames = [os.path.join(other_dir, filename) for filename in sorted(os.listdir(other_dir))]
''' 正负均衡 '''
random.shuffle(tunaju_filename)
random.shuffle(other_filenames)
other_filenames = other_filenames[:2*len(tunaju_filename)]
print(len(other_filenames), len(tunaju_filename))
''' 所有样本名 '''
all_filenames = tunaju_filename + other_filenames
random.shuffle(all_filenames) # 打乱
print(len(all_filenames))

''' 7:train   2:val  1:test 分割样本集 '''
train_flienames, test_filename = train_test_split(all_filenames, train_size=0.9, test_size=0.1)
train_fliename, val_filename = train_test_split(train_flienames, train_size=0.78, test_size=0.22)

model = Net().to(DEVICE)
# model_name = param.PREDICT_MODEL_NAME
# model = torch.load('/home/baiyu/Projects/Sem_image_segmentation/output/models_1/model_4.pt')
optimizer = getattr(torch.optim, param.TRAIN721_OPTIMIZER_NAME)(model.parameters(), lr=LR)

dataset_train = PatchDataset(filenames=train_flienames)
dataset_val = PatchDataset(filenames=val_filename)
dataset_test = PatchDataset(filenames=test_filename)
train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE)


''' *************************训练并保存模型************************* '''
for epoch in range(1, EPOCH + 1):
    logger.info('Epoch:{:d}'.format(epoch))
    train(model, DEVICE, train_loader, optimizer,logger)
    torch.save(model, model_outpath+'/model_{}.pt'.format(epoch))
    val(model, DEVICE, val_loader,logger)

logger.info('.............................Test:...........................')
test(model, DEVICE, test_loader,logger)