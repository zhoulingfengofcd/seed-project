import nets
from nets.deeplab import deeplab
from nets.model import load_model_from_json1


def create_net(in_channels, out_channels, net_name='unet', **kwargs):
    """
    创建网络
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param net_name: 网络类型，可选 unet | deeplabv3 | yolov3
    :return: 创建的网络模型
    """
    if net_name == 'unet':
        net = nets
    elif net_name == 'deeplabv3':
        net = deeplab.DeepLab(num_classes=out_channels, **kwargs)
    elif net_name == 'yolov3':
        if "yolov3_model_json" in kwargs:
            net = load_model_from_json1(kwargs['yolov3_model_json'], in_channels=in_channels)
        else:
            raise Exception("Yolov3 must define the model configuration file `yolov3_model_json` parameter")
    else:
        raise ValueError("Not supported net_name:{}".format(net_name))
    return net


# def ajust_learning_rate(optimizer, lr_strategy, epoch, iteration, epoch_size):
#     """
#     调整学习率
#     :param optimizer: 优化器
#     :param lr_strategy: 策略，一个二维数组，第一位维度对应epoch, 第二维度表示在一个epoch内，若干阶段的学习率
#     :param epoch: 当前在第几个epoch
#     :param iteration: 当前的epoch内，第几次迭代
#     :param epoch_size: 当前的epoch的总迭代次数
#     :return:
#     """

def ajust_learning_rate(optimizer, lr_strategy, epoch, iteration, epoch_size):
    """
    根据给定的策略调整学习率
    @param optimizer: 优化器
    @param lr_strategy: 策略，一个二维数组，第一维度对应epoch，第二维度表示在一个epoch内，若干阶段的学习率
    @param epoch: 当前在第几号epoch
    @param iteration: 当前epoch内的第几次迭代
    @param epoch_size: 当前epoch的总迭代次数
    """
    assert epoch < len(lr_strategy), 'lr strategy unconvering all epoch'
    batch = epoch_size // len(lr_strategy[epoch])
    lr = lr_strategy[epoch][-1]
    for i in range(len(lr_strategy[epoch])):
        if iteration < (i + 1) * batch:
            lr = lr_strategy[epoch][i]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            break
    return lr