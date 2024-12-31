'''
    定义了train,val,test过程
'''
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
from metrics.Iou import iou_numpy
import numpy as np


''' *************************定义训练过程************************* '''
def train(model, device, train_loader, optimizer, logger):
    model.train()
    loss = 0
    batch_num = 0
    for data, target in tqdm(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        batch_num += 1
        if batch_num % 100 ==0:
            logger.info('batch num: {:d}, train Loss: {:.6f}'.format(batch_num, loss.item()))

''' *************************测试过程************************* '''
def val(model, device, test_loader, logger):
    model.eval()
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    y_score = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_true += target.cpu().numpy().tolist()
            y_pred += pred.view_as(target).cpu().numpy().tolist()
            # y_score += np.max(output.cpu().numpy(), axis=1).tolist()
            # y_score = np.vstack((y_score, np.max(output.cpu().numpy())))
    test_loss /= len(test_loader.dataset)
    logger.info('Val loss: {:.4f},  Acc: {:.2f}%,  precision: {:.2f}%,  recall: {:.2f}%'.format(
                test_loss, 100*accuracy_score(y_true, y_pred), 100*precision_score(y_true, y_pred), 100*recall_score(y_true, y_pred)))

''' *************************测试过程************************* '''
def test(model, device, test_loader, logger):
    model.eval()
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    y_score = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_true += target.cpu().numpy().tolist()
            y_pred += pred.view_as(target).cpu().numpy().tolist()
            # y_score += np.max(output.cpu().numpy(), axis=1).tolist()
            # y_score = np.vstack((y_score, np.max(output.cpu().numpy())))
    test_loss /= len(test_loader.dataset)
    logger.info('Test loss: {:.4f},  Acc: {:.2f}%,  precision: {:.2f}%,  recall: {:.2f}%, iou: {:.2f}%'.format(
                test_loss, 100*accuracy_score(y_true, y_pred), 100*precision_score(y_true, y_pred), 100*recall_score(y_true, y_pred), 100*iou_numpy(np.array(y_pred), np.array(y_true))))
