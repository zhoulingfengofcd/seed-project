from utils.net import *
from utils.data import *
import pandas as pd
import os
from utils.loss import *


def train(in_channels, out_channels, net_name, lr, csv_path,
          image_root, label_root, batch_size, out_size, crop_offset,
          epoch_begin, epoch_num,
          num_classes,
          net_state_path):
    """
    训练网络模型
    :param in_channels: 输入通道
    :param out_channels: 输出通道
    :param net_name: 网络名称
    :param lr: 学习率
    :param csv_path: 数据data_list文件
    :param image_root: 图片存放根目录
    :param label_root: 标签根目录
    :param batch_size: 批量尺寸
    :param out_size: 网络输入的图片尺寸
    :param crop_offset: 剪切偏移量
    :param epoch_begin: 开始批次（可以实现断点训练）
    :param epoch_num: epoch大小
    :param num_classes: 类别数量
    :param net_state_path: 训练网络参数保存路径
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 网络
    net = create_net(in_channels, out_channels, net_name)
    # model.train()  # 启用 BatchNormalization 和 Dropout
    # model.eval()  # 不启用 BatchNormalization 和 Dropout, see https://pytorch.org/docs/stable/nn.html?highlight=module%20eval#torch.nn.Module.eval
    net.to(device)

    dice_loss_weights = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8]).to(device)
    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 准备数据
    df = pd.read_csv(csv_path)
    generator = data_generator(image_root,
                               np.array(df['image']),
                               label_root,
                               np.array(df['label']),
                               batch_size, out_size, crop_offset)
    # 训练
    epoch_size = int(len(df)/batch_size)  # 1个epoch包含的batch数目
    for epoch in range(epoch_begin, epoch_num):
        epoch_loss = 0.0
        for iter in range(1, epoch_size+1):
            images, labels = next(generator)
            images = images.to(device)
            labels = labels.to(device)
            # 学习率调整
            learn_rate = 0.001
            predicts = net(images)  # 推断
            optimizer.zero_grad()  # 梯度清零
            loss = create_ce_loss(predicts, labels, num_classes)  # 损失

            loss += create_dice_loss(predicts, labels, num_classes, dice_loss_weights)
            loss += create_miou_loss(predicts, labels, num_classes)
            print("loss {}/{}".format(iter, epoch_size), loss)
            epoch_loss += loss.item()
            loss.backward()  # 反向传播
            optimizer.step()  # 更新网络参数

        # scheduler.step()  # 更新学习率
        # 保存模型
        temp_model_name = f"ckpt_%d_%.2f.pth" % (epoch, epoch_loss)
        ckpt_name = os.path.join(net_state_path, temp_model_name)
        torch.save(net.state_dict(), ckpt_name)
