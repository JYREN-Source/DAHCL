import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1   = nn.BatchNorm1d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes, stride=1)
        self.bn2   = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class ResNet1D(nn.Module):
    """
    经典 ResNet18 结构的 1D 版本（作为特征提取器）
    Stem: 7x1 conv s=2 + maxpool k=3 s=2
    Layers: [2,2,2,2], channels: [64,128,256,512], strides: [1,2,2,2]
    输出: (B, 512, L_out)  不做全局池化，不含分类头
    """
    def __init__(self, block, layers, in_chans=1):
        super().__init__()
        self.inplanes = 64

        # Stem
        self.conv1 = nn.Conv1d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):  # x: (B, C_in, L)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)  # 64
        x = self.layer2(x)  # 128
        x = self.layer3(x)  # 256
        x = self.layer4(x)  # 512
        return x             # (B, 512, L_out)

class FE_ResNet18_1D(ResNet1D):
    def __init__(self, in_chans=1):
        super().__init__(block=BasicBlock1D, layers=[2, 2, 2, 2], in_chans=in_chans)

# ---------------------------
# 分类器与整体模型
# ---------------------------
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.mean(dim=-1)  # global average pooling over temporal dim
        return self.fc(x)

class Model(nn.Module):
    def __init__(self, num_classes=10, in_chans=1):
        super().__init__()
        self.feature_extractor = FE_ResNet18_1D(in_chans=in_chans)  # 输出 (B, 512, L_out)
        self.classifier = Classifier(input_dim=512, output_dim=num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)  # (B, 512, L_out)
        logits = self.classifier(features)
        return logits