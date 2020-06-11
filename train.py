from utils.net import *
from utils.data import *
import pandas as pd
import os


def train(in_channels, out_channels, net_name, lr, csv_path,
          image_root, label_root, batch_size, out_size, crop_offset,
          epoch_begin, epoch_num,
          num_classes,
          net_state_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 网络
    net = create_net(in_channels, out_channels, net_name)
    # model.train()  # 启用 BatchNormalization 和 Dropout
    # model.eval()  # 不启用 BatchNormalization 和 Dropout, see https://pytorch.org/docs/stable/nn.html?highlight=module%20eval#torch.nn.Module.eval
    net.to(device)
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
            loss = create_loss(predicts, labels, num_classes)  # 损失
            print("loss {}/{}".format(iter, epoch_size), loss)
            epoch_loss += loss.item()
            loss.backward()  # 反向传播
            optimizer.step()  # 更新网络参数

        # scheduler.step()  # 更新学习率
        # 保存模型
        temp_model_name = f"ckpt_%d_%.2f.pth" % (epoch, epoch_loss)
        ckpt_name = os.path.join(net_state_path, temp_model_name)
        torch.save(net.state_dict(), ckpt_name)


if __name__ == '__main__':
    train(in_channels=3, out_channels=8, net_name="deeplab", lr=0.001, csv_path="./data_list/train1.csv",
          image_root=r"D:\AI\project\data\baidu_lane_line\original",
          label_root=r"D:\AI\project\data\baidu_lane_line\original",
          batch_size=4, out_size=(224, 224), crop_offset=(0, 0),
          epoch_begin=0, epoch_num=1,
          num_classes=8,
          net_state_path="./outputs/model"
          )
