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
    net.train()  # 启用 BatchNormalization 和 Dropout
    # net.eval()  # 不启用 BatchNormalization 和 Dropout, see https://pytorch.org/docs/stable/nn.html?highlight=module%20eval#torch.nn.Module.eval
    net = net.to(device)
    if load_state_dict_path is not None:
        net.load_state_dict(torch.load(load_state_dict_path))

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


def valid(net, csv_path, load_data, batch_size, resize, crop_offset, num_classes):
    """
    训练网络模型
    :param net: 网络模型
    :param csv_path: 数据data_list文件
    :param load_data: 加载数据function, 返回数据root_path
    :param batch_size: 批量尺寸
    :param resize: 网络输入的图片尺寸
    :param crop_offset: 剪切偏移量
    :param num_classes: 类别数量
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net.eval()  # 不启用 BatchNormalization 和 Dropout, see https://pytorch.org/docs/stable/nn.html?highlight=module%20eval#torch.nn.Module.eval

    # 准备数据
    df = pd.read_csv(csv_path)
    generator = data_generator(load_data,
                               np.array(df['image']),
                               np.array(df['label']),
                               batch_size, resize, crop_offset)
    # 训练
    epoch_size = int(len(df) / batch_size)  # 1个epoch包含的batch数目

    miou = 0.0
    with torch.no_grad():
        for iter in range(1, epoch_size + 1):
            images, labels = next(generator)
            images = images.to(device)
            labels = labels.to(device)

            predicts = net(images)  # 推断

            iou = get_miou(predicts, labels, num_classes)
            print("valid {}/{} iou".format(iter, epoch_size), iou)
            miou += iou
    return miou / epoch_size


def train_valid(in_channels, out_channels, net_name, lr,
                train_csv_path, load_train_data,
                valid_csv_path, load_valid_data,
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
    :param train_csv_path: 数据data_list文件
    :param load_train_data: 加载数据function, 返回数据root_path
    :param valid_csv_path: 数据data_list文件
    :param load_valid_data: 加载数据function, 返回数据root_path
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
    net.train()  # 启用 BatchNormalization 和 Dropout
    # net.eval()  # 不启用 BatchNormalization 和 Dropout, see https://pytorch.org/docs/stable/nn.html?highlight=module%20eval#torch.nn.Module.eval
    net = net.to(device)
    if load_state_dict_path is not None:
        net.load_state_dict(torch.load(load_state_dict_path))

    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1, last_epoch=-1)

    # 准备数据
    df = pd.read_csv(train_csv_path)
    generator = data_generator(load_train_data,
                               np.array(df['image']),
                               np.array(df['label']),
                               batch_size, resize, crop_offset)
    # 训练、验证
    epoch_size = int(len(df) / batch_size)  # 1个epoch包含的batch数目
    best_net = {'miou': 0, 'name': ''}
    for epoch in range(epoch_begin, epoch_num):
        # 训练
        print("The epoch {} start.".format(epoch))
        start = datetime.datetime.now()
        epoch_loss = 0.0
        for batch_index in range(1, epoch_size + 1):
            images, labels = next(generator)
            images = images.to(device)
            labels = labels.to(device)

            predicts = net(images)  # 推断

            if loss_weights is not None:
                # dice_loss_weights = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8]).to(device)
                loss_weights = torch.Tensor(loss_weights).to(device)
            loss = create_multi_loss(loss_type=LossType.ce_loss, predicts=predicts, labels=labels,
                                     num_classes=num_classes, loss_weights=loss_weights)

            print("loss {}/{}".format(batch_index, epoch_size), loss)
            epoch_loss += loss.item()
            loss.backward()  # 反向传播
            optimizer.step()  # 更新网络参数
            optimizer.zero_grad()  # 梯度清零

        print("The epoch {} end, epoch loss:{}, execution time:{}".format(epoch, epoch_loss,
                                                                          datetime.datetime.now() - start))

        # 验证
        miou = valid(net=net, csv_path=valid_csv_path, load_data=load_valid_data, batch_size=batch_size,
                     resize=resize, crop_offset=crop_offset, num_classes=num_classes).item()

        model_name = f"ckpt_%d_%.2f_%.2f.pth" % (epoch, epoch_loss, miou)

        if miou > best_net['miou']:
            best_net['miou'] = miou
            best_net['name'] = model_name

        # 保存模型
        save_model(net, model_name)
        print("The current epoch {} learning rate {}.".format(epoch, scheduler.get_lr()[0]))
        scheduler.step()  # 更新学习率

    print("This is the best model", best_net)


def test(in_channels, out_channels, net_name,
         weights_path, image_path, resize=None, crop_offset=None, **kwargs):
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
        image, ori_size, ori_image = read_image(image_path, resize, crop_offset)
        image = image.to(device)
        predicts = net(image)  # 推断
        convert = torch.softmax(predicts, dim=1).argmax(dim=1)  # convert.shape=(n,h,w)
        # decode_image = decode(convert).permute(1, 2, 0)
        # decode_image = decode_image.expand(decode_image.shape[0], decode_image.shape[1], 3).contiguous()
        # shape=(n,h,w,c)=(n,h,w,3)
        decode_image = decode(convert).unsqueeze(dim=-1). \
            expand(convert.shape[0], convert.shape[1], convert.shape[2], 3).contiguous()
        recover_img = recover_image(decode_image.numpy(), (ori_size[0] - crop_offset[0], ori_size[1] - crop_offset[1]),
                                    crop_offset)

        for index, img in enumerate(recover_img):
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.imshow(ori_image)
            plt.show()


def test1(in_channels, out_channels, net_name,
          weights_path, test_image_root, batch_size, resize=None, crop_offset=None, **kwargs):
    """
    测试网络
    :param in_channels: 输入通道
    :param out_channels: 输出通道
    :param net_name: 网络名称
    :param weights_path: 模型权重文件路径
    :param test_image_root: 测试图片目录
    :param batch_size: 批量大小
    :param resize: 网络输入的图片尺寸
    :param crop_offset: 剪切偏移量
    :param kwargs:
    :return:
    """

    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    # 网络
    net = create_net(in_channels, out_channels, net_name, **kwargs)

    net.eval()  # 不启用 BatchNormalization 和 Dropout, see https://pytorch.org/docs/stable/nn.html?highlight=module%20eval#torch.nn.Module.eval
    net = net.to(device)

    # Load checkpoint weights
    net.load_state_dict(torch.load(weights_path))

    generator, data_size = test_data_generator(test_image_root, batch_size, resize, crop_offset)
    epoch_size = int(data_size / batch_size)  # 1个epoch包含的batch数目
    with torch.no_grad():
        for batch_index in range(1, epoch_size + 1):
            images, original_images, original_hw_size = next(generator)

            images = images.to(device)
            predicts = net(images)  # 推断 shape=(n,c,h,w)
            convert = torch.softmax(predicts, dim=1).argmax(dim=1)  # convert.shape=(n,h,w)

            # shape=(n,h,w,c)=(n,h,w,3)
            # decode_image = decode(convert).unsqueeze(dim=-1). \
            #     expand(convert.shape[0], convert.shape[1], convert.shape[2], 3).contiguous()

            # (n,h,w) => (n,h,w,3)
            decode_image = decode_rgb(convert)

            for index, original_image in enumerate(original_images):
                original_size = original_hw_size[index]
                result_image = recover_image(decode_image[index].numpy(),
                                             (original_size[0] - crop_offset[0], original_size[1] - crop_offset[1]),
                                             crop_offset)

                plt.subplot(1, 2, 1)
                plt.imshow(result_image)
                plt.subplot(1, 2, 2)
                plt.imshow(original_image)
                plt.show()
