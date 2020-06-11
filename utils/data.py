import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import torch
import os
from typing import Tuple

def encoded(labels):
    """
    将标签图的灰度值转换成类别id
    注意：ignoreInEval为True的都当分类0处理
    :param labels:
    :return:
    """
    encoded_labels = np.zeros_like(labels)
    # 除以下像素值，其余像素值属于类别0
    # 1
    encoded_labels[labels == 200] = 1
    encoded_labels[labels == 204] = 1
    encoded_labels[labels == 209] = 1
    # 2
    encoded_labels[labels == 201] = 2
    encoded_labels[labels == 203] = 2
    # 3
    encoded_labels[labels == 217] = 3
    # 4
    # 注意:原本数据集4这个类别ignoreInEval都为True，所以把这个类别往后推了
    encoded_labels[labels == 210] = 4
    # 5
    encoded_labels[labels == 214] = 5
    # 6
    encoded_labels[labels == 220] = 6
    encoded_labels[labels == 221] = 6
    encoded_labels[labels == 222] = 6
    encoded_labels[labels == 224] = 6
    encoded_labels[labels == 225] = 6
    encoded_labels[labels == 226] = 6
    # 7
    encoded_labels[labels == 205] = 7
    encoded_labels[labels == 227] = 7
    encoded_labels[labels == 250] = 7
    return encoded_labels


def decode(labels):
    """
    将类别id恢复成灰度值
    :param labels: shape=(h,w)
    :return:
    """
    decoded_labels = np.zeros_like(labels, dtype=np.int8)
    # 1
    decoded_labels[labels == 1] = 204
    # 2
    decoded_labels[labels == 2] = 203
    # 3
    decoded_labels[labels == 3] = 217
    # 4
    decoded_labels[labels == 4] = 210
    # 5
    decoded_labels[labels == 5] = 214
    # 6
    decoded_labels[labels == 6] = 224
    # 7
    decoded_labels[labels == 7] = 227
    return decoded_labels


def crop_resize(image, label, out_size: Tuple, crop_offset: Tuple = (0, 0)):
    """
    裁剪图片
    :param image: 输入图片,shape=(h,w,c)
    :param label: 标签图片,shape=(h,w)
    :param out_size: 输出图片尺寸，(h,w)
    :param crop_offset: 截掉多少，(start_height, start_width)
    :return: roi_image=shape(h,w,c), roi_label=shape(h,w)
    """
    out_size = out_size[::-1]  # (h,w)转(w,h),resize格式需要(w,h)
    roi_image = image[crop_offset[0]:, crop_offset[1]:]  # crop
    # interpolation - 插值方法。共有5种：
    # １)INTER_NEAREST - 最近邻插值法
    # ２)INTER_LINEAR - 双线性插值法（默认）
    # ３)INTER_AREA - 基于局部像素的重采样（resampling using pixel area relation）。对于图像抽取（image decimation）来说，
    # 这可能是一个更好的方法。但如果是放大图像时，它和最近邻法的效果类似。
    # ４)INTER_CUBIC - 基于4x4像素邻域的3次插值法
    # ５)INTER_LANCZOS4 - 基于8x8像素邻域的Lanczos插值
    roi_image = cv2.resize(roi_image, out_size, interpolation=cv2.INTER_LINEAR)
    if label is not None:
        roi_label = label[crop_offset[0]:, crop_offset[1]:]
        roi_label = cv2.resize(roi_label, out_size, interpolation=cv2.INTER_NEAREST)
    else:
        roi_label = None
    return roi_image, roi_label


def data_generator(image_root, image_list, label_root, label_list, batch_size, out_size: Tuple,
                   crop_offset: Tuple = (0, 0)):
    """
    生成训练数据
    :param image_root: 图片文件的数据集绝对地址
    :param image_list: 图片文件的数据集地址
    :param label_root: 标签文件的数据集绝对地址
    :param label_list: 标签文件的数据集地址
    :param batch_size: 每批取多少张图片
    :param out_size: 输出的图片尺寸，(h,w)
    :param crop_offset: 将原始图片截掉多少，(start_height, start_width)
    :return: out_images=shape(batch_size,c,h,w),out_labels=shape(batch_size,h,w)
    """
    indices = np.arange(0, len(image_list))  # 索引
    out_images = []
    out_labels = []
    out_images_filename = []
    while True:
        np.random.shuffle(indices)
        for i in indices:
            try:
                image = cv2.imread(os.path.join(image_root, image_list[i]))  # (h,w,c) BGR格式
                label = cv2.imread(os.path.join(label_root, label_list[i]), cv2.IMREAD_GRAYSCALE)  # (h,w)
            except:
                continue
            # crop & resize
            image, label = crop_resize(image, label, out_size, crop_offset)  # image=shape(h,w,c), label=shape(h,w)
            # encode
            label = encoded(label)

            out_images.append(image)
            out_labels.append(label)
            out_images_filename.append(image_list[i])
            if len(out_images) == batch_size:
                out_images = np.array(out_images, dtype=np.float32)
                out_labels = np.array(out_labels, dtype=np.int64)
                # BGR转换成RGB
                out_images = out_images[:, :, :, ::-1]
                # 维度改成(n,c,h,w)
                out_images = out_images.transpose(0, 3, 1, 2)
                # 归一化 -1 ~ 1
                out_images = out_images*2/255 - 1
                yield torch.from_numpy(out_images), torch.from_numpy(out_labels).long()
                out_images = []
                out_labels = []
                out_images_filename = []


if __name__ == "__main__":
    # img = Image.open(
    #     r"D:\AI\project\data\baidu_lane_line\original\Gray_Label/Label_road04/Label\Record003\Camera 5\171206_054538078_Camera_5_bin.png"
    # )
    # lab = transforms.ToTensor()(img)
    # plt.imshow(img)
    # plt.show()
    # encoded(lab)

    import pandas as pd
    df = pd.read_csv("../data_list/train.csv")
    generator = data_generator(r"D:\AI\project\data\baidu_lane_line\original",
                               np.array(df['image']),
                               r"D:\AI\project\data\baidu_lane_line\original",
                               np.array(df['label']),
                               2, (200, 416))
    out_imgs, out_labs = next(generator)
    print(out_imgs, out_labs)
