"""
@description: 遍历数据集，提取image跟label的路径，以csv的形式保存下来
"""
import os
import pandas as pd
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.utils import shuffle


def baidu_lane_line(root, train_rate, train_output, valid_output):
    """
    百度车道线检测数据划分
    :param root: 数据存放目录，按照百度官方比赛目录存放，应包括Gray_Label、Road02、Road03、Road04几个目录
    :param train_rate: 训练数据集比例，如0.8，则训练集与验证集8:2
    :param train_output: 训练集csv输出文件存放路径
    :param valid_output: 验证集csv输出文件存放路径
    :return: 无
    """
    # image dir example
    # root + "Road02\ColorImage_road02\ColorImage\Record001\Camera 5"
    # root + "Road02\ColorImage_road02\ColorImage\Record007\Camera 6"
    # root + "Road04\ColorImage_road04\ColorImage\Record001\Camera 5"
    # label dir example
    # root + "Gray_Label\Label_road02\Label\Record001\Camera 5"
    # root + "Gray_Label\Label_road02\Label\Record007\Camera 6"
    # root + "Gray_Label\Label_road04\Label\Record001\Camera 5"
    image_list = []
    label_list = []
    image_dirs = ["Road02/ColorImage_road02/ColorImage",
                  "Road03/ColorImage_road03/ColorImage",
                  "Road04/ColorImage_road04/ColorImage"]
    label_dirs = ["Gray_Label/Label_road02/Label",
                  "Gray_Label/Label_road03/Label",
                  "Gray_Label/Label_road04/Label"]

    for d1 in zip(image_dirs, label_dirs):
        i1 = d1[0]
        l1 = d1[1]
        # Road02,Road03,Road04
        if not os.path.isdir(os.path.join(root, i1)):
            raise Exception(i1+"不是一个目录!")
        if not os.path.isdir(os.path.join(root, l1)):
            raise Exception(l1 + "不是一个目录!")
        for d2 in os.listdir(os.path.join(root, i1)):
            # d2 = Record001,Record002,...
            if not os.path.isdir(os.path.join(root, i1, d2)):
                continue
            for d3 in os.listdir(os.path.join(root, i1, d2)):
                # d3 = Camera 5,Camera 6
                if not os.path.isdir(os.path.join(root, i1, d2, d3)):
                    continue
                for d4 in os.listdir(os.path.join(root, i1, d2, d3)):
                    # all data
                    image_path = os.path.join(root, i1, d2, d3, d4)  # image path
                    label_path = os.path.join(root, l1, d2, d3, d4.replace(".jpg", "_bin.png"))  # label path
                    if os.path.isdir(image_path):
                        continue
                    if os.path.exists(image_path) and os.path.exists(label_path):
                        j = 0
                    elif not os.path.exists(image_path):
                        raise Exception(image_path+"文件不存在!")
                    elif not os.path.exists(label_path):
                        raise Exception(label_path + "文件不存在!")
                    image_list.append(os.path.join(i1, d2, d3, d4))
                    label_list.append(os.path.join(l1, d2, d3, d4.replace(".jpg", "_bin.png")))
    print("总数据", len(image_list), len(label_list))
    shuffle_image, shuffle_label = shuffle(image_list, label_list)  # 打乱样本
    train_data_size = int(len(shuffle_image) * train_rate)
    print("训练数据", train_data_size)
    # 写训练样本文件
    df = pd.DataFrame({'image': shuffle_image[:train_data_size],
                       'label': shuffle_label[:train_data_size]})
    df.to_csv(train_output, index=False)
    # 写验证样本文件
    df = pd.DataFrame({'image': shuffle_image[train_data_size:len(shuffle_image)],
                       'label': shuffle_label[train_data_size:len(shuffle_image)]})
    df.to_csv(valid_output, index=False)


if __name__ == '__main__':
    baidu_lane_line(r"D:\AI\project\data\baidu_lane_line\original",
                    0.8,
                    "../data_list/train.csv",
                    "../data_list/valid.csv")
