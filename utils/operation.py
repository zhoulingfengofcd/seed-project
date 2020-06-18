from utils.net import *
from utils.data import *
import pandas as pd
from utils.loss import *
import datetime

try:
    import moxing as mox
except:
    print('not use moxing')


def train(in_channels, out_channels, net_name, lr, csv_path, load_data,
          batch_size, resize, crop_offset,
          epoch_begin, epoch_num,
          num_classes,
          save_model, **kwargs):
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
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 网络
    net = create_net(in_channels, out_channels, net_name, **kwargs)
    net.train()  # 启用 BatchNormalization 和 Dropout
    # net.eval()  # 不启用 BatchNormalization 和 Dropout, see https://pytorch.org/docs/stable/nn.html?highlight=module%20eval#torch.nn.Module.eval
    net.to(device)

    dice_loss_weights = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8]).to(device)
    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1, last_epoch=-1)

    # 准备数据
    df = pd.read_csv(csv_path)
    generator = data_generator(load_data,
                               np.array(df['image']),
                               np.array(df['label']),
                               batch_size, resize, crop_offset)
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

            # loss = create_ce_loss(predicts, labels, num_classes)  # bce loss
            loss = create_dice_loss(predicts, labels, num_classes, None)  # dice loss
            # loss += create_iou_loss(predicts, labels, num_classes)
            miou = get_miou(predicts, labels, num_classes)  # miou loss
            print("loss {}/{}".format(iter, epoch_size), loss)
            epoch_loss += loss.item()
            loss.backward()  # 反向传播
            optimizer.step()  # 更新网络参数
            optimizer.zero_grad()  # 梯度清零

        print("The epoch {} end, epoch loss:{}, miou:{}, execution time:{}".format(epoch, epoch_loss, miou,
                                                                                   datetime.datetime.now() - start))
        scheduler.step()  # 更新学习率
        # 保存模型
        model_name = f"ckpt_%d_%.2f.pth" % (epoch, epoch_loss)
        save_model(net, model_name)


def test(in_channels, out_channels, net_name,
         weights_path, image_path, resize, crop_offset, **kwargs):
    """
    测试网络
    :param in_channels: 输入通道
    :param out_channels: 输出通道
    :param net_name: 网络名称
    :param weights_path: 模型权重文件路径
    :param image_path: 测试图片地址
    :param resize: 网络输入的图片尺寸
    :param crop_offset: 剪切偏移量
    :param kwargs:
    :return:
    """

    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    # 网络
    net = create_net(in_channels, out_channels, net_name, **kwargs)
    # net.train()  # 启用 BatchNormalization 和 Dropout
    net.eval()  # 不启用 BatchNormalization 和 Dropout, see https://pytorch.org/docs/stable/nn.html?highlight=module%20eval#torch.nn.Module.eval
    net = net.to(device)
    # Load checkpoint weights
    net.load_state_dict(torch.load(weights_path))
    with torch.no_grad():
        image = read_image(image_path, resize, crop_offset)
        image = image.to(device)
        predicts = net(image)  # 推断
        convert = predicts.permute((0, 2, 3, 1)).argmax(dim=3)
        plt.imshow(decode(convert[0]))
        plt.show()
