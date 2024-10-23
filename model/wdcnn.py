import torch
from torch import nn
import utils
from einops import rearrange
class WDCNN(nn.Module):
    def __init__(self,  in_channel=1, out_channel=10):
        super(WDCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )  # 32, 12,12     (24-2) /2 +1

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )  # 32, 12,12     (24-2) /2 +1

        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
            # nn.AdaptiveMaxPool1d(4)
        )  # 32, 12,12     (24-2) /2 +1

        self.fc = nn.Sequential(
            nn.Linear(64, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, out_channel)
        )

    def forward(self, x):
        # print('1',x.shape)
        x =x.view(x.size(0), 1, 1024)
        # x = rearrange(x, 'b l -> b 1 l')
        print(x.shape)
        x = self.layer1(x)  # [16 64]
        # print(x.shape)
        x = self.layer2(x)  # [32 124]
        print(x.shape)
        x = self.layer3(x)  # [64 61]
        # print(x.shape)
        x = self.layer4(x)  # [64 29]
        # print(x.shape)
        x = self.layer5(x)  # [64 13]
        # print('...',x.shape)
        x = x.view(x.size(0), -1)
        # x = x.view(self.batch_size, -1)
        # print('...', x.shape)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    input = torch.randn(20,1024)
    model =WDCNN(in_channel=1, out_channel=10)
    x = model(input)
    print(x.size())
