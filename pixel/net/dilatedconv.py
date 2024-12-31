import torch
import torch.nn as nn
import torch.nn.functional as F
from config import param

INPUT_CHANNEL = param.MODEL_INCHANNEL

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=INPUT_CHANNEL, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(16, affine=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(32, affine=True),

            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dilation=2),
            # nn.ReLU(),
            # nn.BatchNorm2d(64, affine=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64, affine=True),
        )

        self.fc = nn.Linear(5*5*64, 128) # 25
        self.bn4 = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(128, 32)  # 25
        self.fc1_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):

        x = self.conv1(x)


        x = x.view(-1, 5 * 5 * 64) #25
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