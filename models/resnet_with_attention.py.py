import torch
import torch.nn as nn
import torch.nn.functional as F

# ... (其他导入和类定义保持不变)

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W, H = x.size()
        query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)
        key = self.key_conv(x).view(B, -1, W * H)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(x).view(B, -1, W * H)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        return self.gamma * out + x

class ResNet2d(nn.Module):
    def __init__(self, block, n_blocks, keep_prob=1.0, avg_pool=False, drop_rate=0.0,
                 dropblock_size=5, num_classes=-1, use_se=False):
        super(ResNet2d, self).__init__()
        
        # ... (其他初始化代码保持不变)
        
        self.attention1 = SelfAttention(64)
        self.attention2 = SelfAttention(160)
        self.attention3 = SelfAttention(320)
        self.attention4 = SelfAttention(640)

    def forward(self, x):
        x = self.layer1(x)
        x = self.attention1(x)
        x = self.layer2(x)
        x = self.attention2(x)
        x = self.layer3(x)
        x = self.attention3(x)
        x = self.layer4(x)
        x = self.attention4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

# ... (其他函数和类保持不变)
