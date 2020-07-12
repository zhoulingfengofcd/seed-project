from utils.net import *
from utils.data import *
import pandas as pd
from utils.loss import *
import datetime


def train(in_channels, out_channels, net_name, lr, csv_path, data_path,
          batch_size, resize, crop_offset,
          epoch_begin, epoch_num,
          num_classes,
          save_model, load_state_dict_path=None,
          loss_weights=None,
          **kwargs):
    """
    训练网络模型
    :param in_channels: 输入通道
    :param out_channels: 输出通道
    :param net_name: 网络名称
    :param lr: 学习率
    :param csv_path: 数据data_list文件
    :param load_data: 加载数据function, 返回数据root_path
    :param batch_size: 批量尺寸
    :param resize: 网络输入的图片尺寸
    :param crop_offset: 剪切偏移量
    :param epoch_begin: 开始批次（可以实现断点训练）
    :param epoch_num: epoch大小
    :param num_classes: 类别数量
    :param save_model: 训练网络参数保存function
    :param load_state_dict_path: 网络预训练权重
    :param loss_weights: 每个类别的权重, shape=(num_classes)
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 网络
    net = create_net(in_channels, out_channels, net_name, **kwargs)
    net.train()  # 训练 BatchNormalization 和 Dropout
    # net.eval()  # 固定 BatchNormalization 和 Dropout, see https://pytorch.org/docs/stable/nn.html?highlight=module%20eval#torch.nn.Module.eval
    net = net.to(device)
    if load_state_dict_path is not None:
        net.load_state_dict(torch.load(load_state_dict_path))

    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1, last_epoch=-1)

    # 准备数据
    df = pd.read_csv(csv_path)
    generator = hpmp_data_generator(data_path, batch_size)
    # 训练
    epoch_size = int(len(df) / batch_size)  # 1个epoch包含的batch数目
    for epoch in range(epoch_begin, epoch_num):
        print("The epoch {} start.".format(epoch))
        start = datetime.datetime.now()
        epoch_loss = 0.0
        for iter in range(1, epoch_size + 1):
            images, labels = next(generator)
            images = images.to(device)
            labels = labels.to(device)

            predicts = net(images)  # 推断

            if loss_weights is not None:
                # dice_loss_weights = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8]).to(device)
                loss_weights = torch.Tensor(loss_weights).to(device)
            loss = create_multi_loss(loss_type=LossType.ce_loss, predicts=predicts, labels=labels,
                                     num_classes=num_classes, loss_weights=loss_weights)

            miou = get_miou(predicts, labels, num_classes)

            print("loss {}/{}".format(iter, epoch_size), loss)
            epoch_loss += loss.item()
            loss.backward()  # 反向传播
            optimizer.step()  # 更新网络参数
            optimizer.zero_grad()  # 梯度清零

        print("The epoch {} end, epoch loss:{}, miou:{}, execution time:{}".format(epoch, epoch_loss, miou.item(),
                                                                                   datetime.datetime.now() - start))
        print("The current epoch {} learning rate {}.".format(epoch, scheduler.get_lr()[0]))
        scheduler.step()  # 更新学习率
        # 保存模型
        model_name = f"ckpt_%d_%.2f.pth" % (epoch, epoch_loss)
        save_model(net, model_name)