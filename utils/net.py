import nets
from nets.deeplab import deeplab
import torch
import torch.nn.functional as F


def create_net(in_channels, out_channels, net_name='unet'):
    """
    创建网络
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param net_name: 网络类型，可选 unet | deeplab
    :return: 创建的网络模型
    """
    if net_name == 'unet':
        net = nets
    elif net_name == 'deeplab':
        net = deeplab.DeepLab(num_classes=out_channels)
    else:
        raise ValueError("Not supported net_name:{}".format(net_name))
    return net


def create_loss(predicts: torch.Tensor, labels: torch.Tensor, num_classes):
    """
    创建loss
    :param predicts: shape=(n,c,h,w)
    :param labels: shape=(n,h,w) or shape=(n,1,h,w)
    :param num_classes: int should equal to channels of predicts
    :return: loss, mean_iou
    """
    # 维度换位(n,h,w,c)
    predicts = predicts.permute((0, 2, 3, 1))
    # reshape to (-1, num_classes) 每个像素在每种分类上都有一个概率
    predicts = predicts.reshape((-1, num_classes))
    # bce with dice
    ce_loss = F.cross_entropy(input=predicts, target=labels.flatten())  # log(softmax(input)) -> NLLLoss(input, target)
    return ce_loss


def ajust_learning_rate(optimizer, lr_strategy, epoch, iteration, epoch_size):
    """
    调整学习率
    :param optimizer: 优化器
    :param lr_strategy: 策略，一个二维数组，第一位维度对应epoch, 第二维度表示在一个epoch内，若干阶段的学习率
    :param epoch: 当前在第几个epoch
    :param iteration: 当前的epoch内，第几次迭代
    :param epoch_size: 当前的epoch的总迭代次数
    :return:
    """