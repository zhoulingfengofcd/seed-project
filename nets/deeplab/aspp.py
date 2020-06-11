import torch
import torch.nn as nn
import torch.nn.functional as F


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()

        # out = (n + 2p-((d-1) * (f - 1) + f)) / s + 1 = (n+2p-(df-d+1))/s+1
        # 如果p=d，out = (n+3p-pf-1)/s+1
        # 再如果s=1，out = n+3p-pf
        # 再如果f=3，out = n
        # 也就是，如果f=3，s=1，p=d，那么out=n，输出尺寸不变

        # aspp1,filter=1,padding=0,dilation=1,输入2048*28*28,out = (28+0-1)/1+1 = 28
        # aspp2, filter = 3, padding = 12, dilation = 12,输入2048*28*28,out = (28+24-25)/1+1 = 28
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, output_stride):
        super(ASPP, self).__init__()
        inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        BatchNorm = nn.BatchNorm2d

        # 假设 output_stride = 8
        # aspp1,filter=3,padding=0,dilation=1,输入2048*28*28，输出256*28*28
        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        # aspp2,filter=3,padding=12,dilation=12,输入2048*28*28，输出256*28*28
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        # aspp3,filter=3,padding=24,dilation=24,输入2048*28*28，输出256*28*28
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        # aspp4,filter=3,padding=36,dilation=36,输入2048*28*28，输出256*28*28
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 输入2048*28*28，输出2048*1*1
            nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),  # 输入2048*1*1，输出256*1*1
            BatchNorm(256),
            nn.ReLU()
        )
        # 输入通道数=256*4+256=1280,即输入1280*28*28，输出256*28*28
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()