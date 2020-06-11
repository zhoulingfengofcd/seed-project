import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.deeplab import resnet
from nets.deeplab import aspp
from nets.deeplab import decoder
from utils import visual
# 参考链接：https://github.com/jfzhang95/pytorch-deeplab-xception


class DeepLab(nn.Module):
    def __init__(self,  output_stride=8, num_classes=21):
        super(DeepLab, self).__init__()

        self.backbone = resnet.ResNet101(output_stride)
        self.aspp = aspp.ASPP(output_stride)
        self.decoder = decoder.Decoder(num_classes)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)  # 输出num_classes*56*56
        # print(x.shape)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # 按照输入图片尺寸上采样

        return x


if __name__ == "__main__":
    model = DeepLab(output_stride=8)
    model.eval()
    input = torch.rand(1, 3, 224, 224)
    output = model(input)
    print(output.size())
    g = visual.make_dot(output)
    g.view()