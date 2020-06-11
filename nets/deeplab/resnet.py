import torch.nn as nn
import torch
import torchvision
import torch.utils.model_zoo as model_zoo
import math
# 参考上课代码和官方pytorch resnet代码，https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)  # ? Why no bias: 如果卷积层之后是BN层，那么可以不用偏置参数，可以节省内存


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation, padding=padding, bias=False)  # ? Why no bias


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer=None):
        """
        :param inplanes: 该块(Bottleneck)输入通道数
        :param planes: 该块所在层的输入通道数
        :param stride: 3*3卷积步长
        :param downsample:
        :param norm_layer:
        """
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # (n+2p-f)/s + 1 = (n-1)/1+1 = n
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        # stride=1,dilation=padding=1 (n+2p-f)/s + 1 = (n+2-3)/1+1 = n
        # stride=1,dilation=padding=2 (n+2p-((d-1)*(f-1)+f))/s + 1 = (n+4-5)/1+1 = n
        # stride=1,dilation=padding=4 (n+2p-((d-1)*(f-1)+f))/s + 1 = (n+8-9)/1+1 = n
        # stride=1,dilation=padding=12 (n+2p-((d-1)*(f-1)+f))/s + 1 = (n+24-25)/1+1 = n
        # stride=2,dilation=padding=1 (n+2-3)/2+1 = (n-1)/2+1
        # stride=2,dilation=padding=2 (n+4-5)/2+1 = (n-1)/2+1
        self.conv2 = conv3x3(planes, planes, stride, dilation=dilation, padding=dilation)
        self.bn2 = norm_layer(planes)
        # (n+2p-f)/s + 1 = (n-1)/1+1 = n
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet101(nn.Module):
    def __init__(self, output_stride):
        """
        :param num_class:输出类别数量
        """
        super(ResNet101, self).__init__()
        self.inplanes = 64  # 记录每个layer的channel

        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # 假设 output_stride = 8
        # out = (n+2p-f)/s + 1
        # 输入3*224*224，宽高(224+2*3-7)/2+1=112，输出64*112*112
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # (112+2-3)/2+1=56，输出64*56*56
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layer1,stride=1,dilation=1,输入64*56*56，输出256*56*56
        self.layer1 = self._make_layer(64, 3, stride=strides[0], dilation=dilations[0])  # 50层及以上resnet使用bottleneck块，输入64通道，3个block
        # layer2,stride=2,dilation=1,输入256*56*56，输出512*28*28
        self.layer2 = self._make_layer(128, 4, stride=strides[1], dilation=dilations[1])
        # layer3,stride=1,dilation=2,输入512*28*28，输出1024*28*28
        self.layer3 = self._make_layer(256, 23, stride=strides[2], dilation=dilations[2])
        # layer4,stride=1,dilation=4,输入1024*28*28，输出2048*28*28
        self.layer4 = self._make_MG_unit(512, [3, 4, 23, 3], stride=strides[3], dilation=dilations[3])

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)等于GAP
        # self.fc = nn.Linear(512 * 4, num_class)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1, dilation=1):
        """
        :param planes: 输入通道数
        :param blocks: blocks的数量
        :param stride: 第一个块3*3卷积层的步长
        :return:
        """
        downsample = None

        if stride != 1 or self.inplanes != planes * 4:
            # 需要调整维度
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * 4, stride),  # 同时调整spatial(H x W))和channel两个方向
                nn.BatchNorm2d(planes * 4)
            )
        layers = []
        # layer1,stride=1,dilation=1,输入64*56*56，输出256*56*56
        # layer2,stride=2,dilation=1,输入256*56*56，输出512*28*28
        # layer3,stride=1,dilation=2,输入512*28*28，输出1024*28*28
        layers.append(Bottleneck(self.inplanes, planes, stride, dilation, downsample, nn.BatchNorm2d))  # 第一个block单独处理

        self.inplanes = planes * 4  # 记录layerN的channel变化，具体请看ppt resnet表格

        for _ in range(1, blocks):  # 从1开始循环，因为第一个模块前面已经单独处理
            # layer1,stride=1,dilation=1,输入256*56*56，输出256*56*56
            # layer2,stride=1,dilation=1,输入512*28*28，输出512*28*28
            # layer3,stride=1,dilation=2,输入1024*28*28，输出1024*28*28
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation, norm_layer=nn.BatchNorm2d))
        return nn.Sequential(*layers)  # 使用Sequential层组合blocks，形成stage。如果layers=[2,3,4]，那么*layers=？

    def _make_MG_unit(self, planes, blocks, stride=1, dilation=1):
        """
        与_make_layer区别在于，dilation赋值多乘了blocks[0]，即dilation=该layer块数量 * dilation
        :param planes: 输入通道数
        :param blocks: blocks的数量
        :param stride: 第一个块3*3卷积层的步长
        :return:
        """
        downsample = None

        if stride != 1 or self.inplanes != planes * 4:
            # 需要调整维度
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * 4, stride),  # 同时调整spatial(H x W))和channel两个方向
                nn.BatchNorm2d(planes * 4)
            )
        layers = []
        # layer4,stride=1,dilation=12,输入1024*28*28，输出2048*28*28
        layers.append(Bottleneck(self.inplanes, planes, stride, blocks[0]*dilation, downsample, nn.BatchNorm2d))  # 第一个block单独处理

        self.inplanes = planes * 4  # 记录layerN的channel变化，具体请看ppt resnet表格

        for _ in range(1, len(blocks)):  # 从1开始循环，因为第一个模块前面已经单独处理
            # layer1,stride=1,dilation=12,输入2048*28*28，输出2048*28*28
            layers.append(Bottleneck(self.inplanes, planes, dilation=blocks[0]*dilation, norm_layer=nn.BatchNorm2d))
        return nn.Sequential(*layers)  # 使用Sequential层组合blocks，形成stage。如果layers=[2,3,4]，那么*layers=？

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 去掉平均池化和全连接层
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


if __name__ == '__main__':
    # 打印核实网络是否与官方的一致
    # model = torchvision.models.resnet101()
    model = ResNet101(10)
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)