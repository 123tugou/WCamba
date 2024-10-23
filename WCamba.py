import torch.nn as nn
from mamba_ssm import Mamba
import torch
from einops import rearrange, repeat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def savedata(tensor,name='tensor.txt'):
# 保存Tensor到txt文件
    with open(name, 'w') as f:
        for row in tensor:
            f.write(' '.join(map(str, row.tolist())) + '\n')

class mambaModel(nn.Module):
    def __init__(self,input_dim, num_layers, output_dim, d_state=16, d_conv=4, expand=2):
        super(mambaModel, self).__init__()


        self.input_dim = input_dim
        self.num_layers = num_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.mamba = Mamba(input_dim//32, d_state, d_conv, expand)
        self.norm = nn.LayerNorm(input_dim//32)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1,32, kernel_size=64, stride=16, padding=24),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            )
        # 平局池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(input_dim//32, output_dim)

    def forward(self, input_seq):   # num_layers = 2   cwru pu xj best model
        out = input_seq.view(input_seq.size(0), 1, self.input_dim)
        # print(out.size())
        # print(out)
        # savedata(out,name='SNRtensor.txt')
        for i in range(self.num_layers):
            out = out.view(out.size(0), 1, self.input_dim) # [32,1,1024]
            out = self.layer1(out) # [32,32,32]
            out = self.mamba(out) + out # [32,32,32]
            out = self.layer2(out) + out # [32,32,32]

            # print('{}'.format(i),out)
            # savedata(out.view(out.size(0), 1, self.input_dim), name=f'SNRtensor{i}.txt')
        out = self.norm(out) # [32,32,32]
        out = self.avgpool(out.transpose(1, 2))  # [32,32,1]
        flat_tensor = out.view(out.size(0), -1)  # [32,32]
        out = self.fc(flat_tensor)
        # print(out.size())
        return out


    # def forward(self, input_seq):   # mamba
    #     out = rearrange(input_seq, 'b l -> b 1 l')
    #
    #     for i in range(self.num_layers):
    #         out = out.view(out.size(0), 32, 32)
    #         out = self.mamba(out)
    #
    #     out = self.norm(out)
    #     out = self.avgpool(out.transpose(1, 2))
    #     flat_tensor = out.view(out.size(0), -1)
    #     out = self.fc(flat_tensor)
    #     # print(out.size())
    #     return out

    #
    # def forward(self, input_seq):   # MAMBA+RES
    #     out = rearrange(input_seq, 'b l -> b 1 l')
    #     # print(out.size())
    #     # print(out)
    #     # savedata(out,name='SNRtensor.txt')
    #     for i in range(self.num_layers):
    #         out = out.view(out.size(0), 32, 32)
    #         out_1 = out
    #         out = self.mamba(out)
    #         out = out+out_1
    #     out = self.norm(out)
    #     out = self.avgpool(out.transpose(1, 2))
    #     flat_tensor = out.view(out.size(0), -1)
    #     out = self.fc(flat_tensor)
    #     # print(out.size())
    #     return out