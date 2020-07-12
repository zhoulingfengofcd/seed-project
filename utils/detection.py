from utils.detection_util import get_yolo_output, non_max_suppression, get_batch_statistics, xywh2xyxy, ap_per_class, \
    xyhw2xyxy
from utils.net import *
from utils.data import *
import pandas as pd
from utils.loss import *
import datetime


def valid(net, csv_path, load_data, batch_size, resize, crop_offset, num_classes,
          load_classes, anchors,
          iou_thres, conf_thres, nms_thres,
          img_size):
    """
    训练网络模型
    :param iou_thres: 在批量数据统计时，当iou值大于该阈值时，才认为是正确的
    :param nms_thres: 非最大值抑制时，预测框相似程度的iou阈值，当大于该阈值，则认为预测框相似，过滤
    :param conf_thres: 非最大值抑制时，小于该置信度阈值，则过滤
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
    float_tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    net.eval()  # 不启用 BatchNormalization 和 Dropout, see https://pytorch.org/docs/stable/nn.html?highlight=module%20eval#torch.nn.Module.eval

    # 准备数据
    df = pd.read_csv(csv_path)
    # generator = data_generator(load_data,
    #                            np.array(df['image']),
    #                            np.array(df['label']),
    #                            batch_size, resize, crop_offset)
    generator = detection_data_generator(load_data,
                                         np.array(df['image']),
                                         np.array(df['label']),
                                         batch_size, load_classes=load_classes, resize=resize, crop_offset=crop_offset)
    # 训练
    epoch_size = int(len(df) / batch_size)  # 1个epoch包含的batch数目

    # miou = 0.0
    targets = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    with torch.no_grad():
        for iter in range(1, epoch_size + 1):
            images, labels = next(generator)
            images = images.to(device)
            # labels = labels.to(device)

            predicts = net(images)  # 推断

            yolo_outputs, _, _ = get_yolo_output(
                predicts=predicts,
                num_classes=num_classes,
                input_size=images.shape[-2:],
                anchors=anchors,
                cuda=torch.cuda.is_available(),
                labels=None,
                # ignore_threshold=0.5,
                # obj_scale=1,
                # noobj_scale=100
            )

            outputs = non_max_suppression(yolo_outputs.detach().cpu(), conf_thres=conf_thres, nms_thres=nms_thres)

            # Extract labels
            targets += labels[:, 1].tolist()
            # Rescale target
            labels[:, 2:] = xyhw2xyxy(labels[:, 2:])
            labels[:, 2:] = labels[:, 2:] * torch.FloatTensor([img_size[1], img_size[0], img_size[1], img_size[0]])
            sample_metrics += get_batch_statistics(outputs, labels, iou_threshold=iou_thres)

            # iou = get_miou(predicts, labels, num_classes)
            # print("valid {}/{} iou".format(iter, epoch_size), iou)
            # miou += iou

    if len(sample_metrics) > 0:
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, targets)

        return precision, recall, AP, f1, ap_class
    else:
        return None


def train_valid(in_channels, out_channels, net_name, lr,
                train_csv_path, load_train_data,
                valid_csv_path, load_valid_data,
                batch_size, resize, crop_offset,
                epoch_begin, epoch_num,
                num_classes,
                load_classes,
                anchors,
                lr_strategy,
                save_model, load_state_dict_path=None,
                loss_type: LossType = LossType.ce_loss, loss_weights=None,
                load_state_dict=None,
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
    :param loss_type: 损失函数
    :param loss_weights: 每个类别的权重, shape=(num_classes)
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 网络
    net = create_net(in_channels, out_channels, net_name, **kwargs)
    net.train()  # 启用 BatchNormalization 和 Dropout
    # net.eval()  # 不启用 BatchNormalization 和 Dropout, see https://pytorch.org/docs/stable/nn.html?highlight=module%20eval#torch.nn.Module.eval
    net = net.to(device)
    if load_state_dict is not None:
        net.load_state_dict(load_state_dict())

    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1, last_epoch=-1)

    # 准备数据
    df = pd.read_csv(train_csv_path)
    generator = detection_data_generator(load_train_data,
                                         np.array(df['image']),
                                         np.array(df['label']),
                                         batch_size, load_classes=load_classes, resize=resize, crop_offset=crop_offset)
    # 训练、验证
    epoch_size = int(len(df) / batch_size)  # 1个epoch包含的batch数目
    best_net = {'mAP': 0, 'name': ''}
    for epoch in range(epoch_begin, epoch_num):
        # 训练
        print("The epoch {} start.".format(epoch))
        start = datetime.datetime.now()
        epoch_loss = 0.0
        for batch_index in range(1, epoch_size + 1):
            images, labels = next(generator)
            images = images.to(device)
            labels = labels.to(device)

            lr = ajust_learning_rate(optimizer, lr_strategy, epoch, batch_index - 1, epoch_size)

            predicts = net(images)  # 推断

            if loss_weights is not None:
                # dice_loss_weights = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8]).to(device)
                loss_weights = torch.Tensor(loss_weights).to(device)

            yolo_outputs, loss, metrics_table = get_yolo_output(
                predicts=predicts,
                num_classes=num_classes,
                input_size=images.shape[-2:],
                anchors=anchors,
                cuda=torch.cuda.is_available(),
                labels=labels,
                ignore_threshold=0.5,
                obj_scale=1,
                noobj_scale=5,
                coord_scale=5,
                cls_scale=1
            )

            print(
                "batch_index/epoch_size/epoch/lr/loss {}/{}/{}/{}/{}".format(batch_index, epoch_size, epoch, lr, loss))
            if metrics_table is not None and len(metrics_table) > 0:
                for index, item in enumerate(metrics_table):
                    keys = item.keys()
                    keys_str = "/".join([key for key in keys if not key == 'grid_size'])
                    values_str = "/".join([str(item[key]) for key in keys if not key == 'grid_size'])
                    print(str(item['grid_size'][
                                  0]) + "/batch_index/epoch_size/epoch/output_loss/" + keys_str + " {}/{}/{}/{}/".format(
                        batch_index, epoch_size, epoch, loss) + values_str)
            # print("batch_index/epoch_size/epoch {}/{}/{}".format(batch_index, epoch_size, epoch), metrics_table)
            epoch_loss += loss.item()
            loss.backward()  # 反向传播
            optimizer.step()  # 更新网络参数
            optimizer.zero_grad()  # 梯度清零

        print("The epoch {} end, epoch loss:{}, execution time:{}".format(epoch, epoch_loss,
                                                                          datetime.datetime.now() - start))

        # 验证
        valid_result = valid(net=net,
                             csv_path=valid_csv_path,
                             load_data=load_valid_data,
                             batch_size=batch_size,
                             resize=resize,
                             crop_offset=crop_offset,
                             num_classes=num_classes,
                             load_classes=load_classes,
                             anchors=anchors,
                             iou_thres=0.5,
                             conf_thres=0.5,
                             nms_thres=0.5,
                             img_size=resize
                             )
        if valid_result is not None:
            precision, recall, ap, f1, ap_class = valid_result

            print("Average Precisions:")
            for i, c in enumerate(ap_class):
                print(f"+ Class '{c}' ({load_classes()[c]}) - AP: {ap[i]}")

            m_ap = ap.mean()
            model_name = f"ckpt_%d_%.2f_%.2f.pth" % (epoch, epoch_loss, m_ap)

            if m_ap > best_net['mAP']:
                best_net['mAP'] = m_ap
                best_net['name'] = model_name

            # 保存模型
            save_model(net, model_name)
            print("The current epoch {} mAP {}.".format(epoch, m_ap))
            # scheduler.step()  # 更新学习率
        else:
            model_name = f"ckpt_%d_%.2f_%.2f.pth" % (epoch, epoch_loss, 0)
            # 保存模型
            save_model(net, model_name)
            print("The current epoch {} mAP {}.".format(epoch, 0))

    print("This is the best model", best_net)
