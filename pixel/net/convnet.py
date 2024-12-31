
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import param

INPUT_CHANNEL = param.MODEL_INCHANNEL

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=INPUT_CHANNEL, out_channels=32, kernel_size=3, padding=(1, 1)),  # 32*25*25
            nn.ReLU(),  # 25*25*32
            nn.Conv2d(32, 32, 3, padding=(1, 1)),  # 25*25*32
            nn.ReLU(),  # 25*25*32
            nn.MaxPool2d((2, 2)),  # 32*12*12
            nn.BatchNorm2d(32, affine=True),  # 32*12*12

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=(1, 1)),  # 64*12*12
            nn.ReLU(),  # 64*12*12
            nn.Conv2d(64, 64, 3, padding=(1, 1)),  # 64*12*12
            nn.ReLU(),  # 64*12*12
            nn.MaxPool2d((2, 2)),  # 64*6*6
            nn.BatchNorm2d(64, affine=True),  # 64*6*6

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=(1, 1)),  # 128*6*6
            nn.ReLU(),  # 128*6*6
            nn.Conv2d(128, 128, 3, padding=(1, 1)),  # 128*6*6
            nn.ReLU(),  # 128*6*6
            nn.MaxPool2d((2, 2)),  # 128*3*3
        )

        self.fc = nn.Linear(3*3*128, 128) # 25
        self.bn4 = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(128, 32)  # 25
        self.fc1_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):

        x = self.conv(x)

        x = x.view(-1, 3*3*128) #25
        x = F.relu(self.fc(x))
        x = self.bn4(x)

        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

#
''' 计算全连接层参数 '''
if __name__ == '__main__':
    model = Net()
    print(model)
    data_input = torch.randn([2, 4, 25, 25])
    print(data_input.size())
    out = model(data_input)
    print(out.size())