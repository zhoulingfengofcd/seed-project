import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import torch
import os
from typing import Tuple

from utils.data_augment import DataAugmentForObjectDetection
from utils.file import *
import warnings
from utils.xml_helper import *


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
    :param labels: shape=(-1)
    :return:
    """
    decoded_labels = torch.zeros_like(labels, dtype=torch.uint8)
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


def decode_rgb(labels):
    """
    将类别id恢复成彩色图
    :param labels: shape=(n,h,w)
    :return: shape=(n,h,w,3)
    """
    decoded_labels = torch.zeros(size=labels.shape + (3,), dtype=torch.uint8)
    # 1
    decoded_labels[labels == 1] = torch.tensor([204, 50, 150], dtype=torch.uint8)
    # 2
    decoded_labels[labels == 2] = torch.tensor([203, 200, 50], dtype=torch.uint8)
    # 3
    decoded_labels[labels == 3] = torch.tensor([217, 200, 0], dtype=torch.uint8)
    # 4
    decoded_labels[labels == 4] = torch.tensor([210, 100, 0], dtype=torch.uint8)
    # 5
    decoded_labels[labels == 5] = torch.tensor([214, 150, 0], dtype=torch.uint8)
    # 6
    decoded_labels[labels == 6] = torch.tensor([224, 0, 50], dtype=torch.uint8)
    # 7
    decoded_labels[labels == 7] = torch.tensor([227, 0, 150], dtype=torch.uint8)
    return decoded_labels


def crop_resize(image, label, resize: Tuple, crop_offset: Tuple = (0, 0)):
    """
    裁剪图片
    :param image: 输入图片,shape=(h,w,c)
    :param label: 标签图片,shape=(h,w)
    :param resize: 输出图片尺寸，(h,w)
    :param crop_offset: 截掉多少，(start_height, start_width)
    :return: roi_image=shape(h,w,c), roi_label=shape(h,w)
    """
    out_size = resize[::-1]  # (h,w)转(w,h),resize格式需要(w,h)
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


def recover_crop_resize(image, label, resize: Tuple, crop_offset: Tuple = (0, 0)):
    """
    恢复裁剪
    :param image: 输入图片,shape=(h,w,c)
    :param label: 标签图片,shape=(h,w)
    :param resize: 输出图片尺寸，(h,w)
    :param crop_offset: 截掉多少，(start_height, start_width)
    :return: roi_image=shape(h,w,c), roi_label=shape(h,w)
    """
    out_size = resize[::-1]  # (h,w)转(w,h),resize格式需要(w,h)
    # interpolation - 插值方法。共有5种：
    # １)INTER_NEAREST - 最近邻插值法
    # ２)INTER_LINEAR - 双线性插值法（默认）
    # ３)INTER_AREA - 基于局部像素的重采样（resampling using pixel area relation）。对于图像抽取（image decimation）来说，
    # 这可能是一个更好的方法。但如果是放大图像时，它和最近邻法的效果类似。
    # ４)INTER_CUBIC - 基于4x4像素邻域的3次插值法
    # ５)INTER_LANCZOS4 - 基于8x8像素邻域的Lanczos插值
    roi_image = cv2.resize(image, out_size, interpolation=cv2.INTER_LINEAR)
    # roi_image = roi_image[crop_offset[0]:, crop_offset[1]:]  # crop
    roi_image = np.pad(roi_image, ((crop_offset[0], 0), (crop_offset[1], 0), (0, 0)), 'constant',
                       constant_values=(0, 0))
    if label is not None:
        roi_label = cv2.resize(label, out_size, interpolation=cv2.INTER_NEAREST)
        # roi_label = roi_label[crop_offset[0]:, crop_offset[1]:]
        roi_label = np.pad(roi_label, ((crop_offset[0], crop_offset[1])), 'constant', constant_values=(0, 0))
    else:
        roi_label = None
    return roi_image, roi_label


def data_generator(load_data, image_list, label_list, batch_size, resize: Tuple,
                   crop_offset: Tuple = (0, 0)):
    """
    生成训练数据
    :param load_data: 加载数据function, 返回数据root_path
    :param image_list: 图片文件的数据集地址
    :param label_list: 标签文件的数据集地址
    :param batch_size: 每批取多少张图片
    :param resize: 输出的图片尺寸，(h,w)
    :param crop_offset: 将原始图片截掉多少，(start_height, start_width)
    :return: out_images=shape(batch_size,c,h,w),out_labels=shape(batch_size,h,w)
    """
    indices = np.arange(0, len(image_list))  # 索引
    out_images = []
    out_labels = []
    out_images_filename = []
    root_path = load_data()
    print("The load data root path %s" % root_path)
    check_path = os.path.join(root_path, image_list[0])
    if not os.path.isfile(check_path):
        print_dir(root_path)
        raise Exception("Check that the data set path `{}` is correct".format(check_path))
    while True:
        np.random.shuffle(indices)
        for i in indices:
            try:
                image = cv2.imread(os.path.join(root_path, image_list[i]))  # (h,w,c) BGR格式
                label = cv2.imread(os.path.join(root_path, label_list[i]), cv2.IMREAD_GRAYSCALE)  # (h,w)
            except:  # 发生异常，执行这块代码
                warnings.warn("The image file `{}` or label file `{}` is no exists".format(
                    os.path.join(root_path, image_list[i]),
                    os.path.join(root_path, label_list[i])))
            else:  # 如果没有异常执行这块代码
                if image is None or label is None:
                    warnings.warn("The image file `{}` or label file `{}` is no exists".format(
                        os.path.join(root_path, image_list[i]),
                        os.path.join(root_path, label_list[i])))
                    continue
                if resize is not None and crop_offset is not None:
                    # crop & resize
                    image, label = crop_resize(image, label, resize,
                                               crop_offset)  # image=shape(h,w,c), label=shape(h,w)
                else:
                    print("The out_size and crop_offset is not set.")
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
                    out_images = out_images * 2 / 255 - 1
                    yield torch.from_numpy(out_images), torch.from_numpy(out_labels).long()  # .requires_grad_(False)
                    # yield torch.from_numpy(out_images), transforms.ToTensor()(out_labels)
                    # yield torch.from_numpy(out_images), Variable(torch.from_numpy(out_labels).long(), requires_grad=False)
                    out_images = []
                    out_labels = []
                    out_images_filename = []


def detection_data_generator(load_data, image_list, label_list, batch_size, load_classes,
                             resize: Tuple,
                             crop_offset: Tuple = (0, 0),
                             is_aug=True
                             ):
    """
    生成训练数据
    :param load_data: 加载数据function, 返回数据root_path
    :param image_list: 图片文件的数据集地址
    :param label_list: 标签文件的数据集地址
    :param batch_size: 每批取多少张图片
    :param classes: 分类类别列表
    :return: out_images=shape(batch_size,c,h,w),
    out_labels=[[batch_size, class_index, x_center, y_center, height, width], ...]
    """
    indices = np.arange(0, len(image_list))  # 索引
    out_images = []
    out_labels = []

    root_path = load_data()
    print("The load data root path %s" % root_path)
    check_path = os.path.join(root_path, image_list[0])
    if not os.path.isfile(check_path):
        print_dir(root_path)
        raise Exception("Check that the data set path `{}` is correct".format(check_path))

    classes = load_classes()
    while True:
        np.random.shuffle(indices)
        for i in indices:
            try:
                image = cv2.imread(os.path.join(root_path, image_list[i]))  # (h,w,c) BGR格式
                # label = cv2.imread(os.path.join(root_path, label_list[i]), cv2.IMREAD_GRAYSCALE)  # (h,w)
                # label = parse_convert_xml(os.path.join(root_path, label_list[i]))  # [[x_center, y_center, height, width, name], ...]
                label = parse_xml(os.path.join(root_path, label_list[i]), list(image.shape[:2]))  # [[x_min, y_min, x_max, y_max, name]]
            except ValueError as e:
                raise e
            except BaseException:  # 发生异常，执行代码
                warnings.warn("The image file `{}` or label file `{}` is no exists".format(
                    os.path.join(root_path, image_list[i]),
                    os.path.join(root_path, label_list[i])))
            else:  # 如果没有异常执行这块代码
                if image is None or label is None:
                    warnings.warn("The image file `{}` or label file `{}` is no exists".format(
                        os.path.join(root_path, image_list[i]),
                        os.path.join(root_path, label_list[i])))
                    continue
                if is_aug:
                    # 数据增强(平移/改变亮度/加噪声/翻转)
                    data_aug = DataAugmentForObjectDetection(crop_rate=0, rotation_rate=0, cutout_rate=0)
                    image, bboxes = data_aug.data_augment(image,
                                                          [coord[:4] for coord in label],
                                                          os.path.join(root_path, image_list[i]),
                                                          os.path.join(root_path, label_list[i])
                                                          )
                    # label[:, :4] = convert_xyxy2xyhw(list(image.shape[:2]), bboxes)
                    label = [item[0]+[item[1][-1]] for item in zip(convert_xyxy2xyhw(list(image.shape[:2]), bboxes), label)]
                else:
                    label = [item[0] + [item[1][-1]] for item in
                             zip(convert_xyxy2xyhw(list(image.shape[:2]), [coord[:4] for coord in label]), label)]

                if resize is not None and crop_offset is not None:
                    # crop & resize
                    image, _ = crop_resize(image, None, resize,
                                           crop_offset)  # image=shape(h,w,c), label=shape(h,w)

                out_images.append(image)
                out_labels = out_labels + [
                    [len(out_images) - 1] + [classes.index(auged_bbox[1][4])] + auged_bbox[0][:4] for auged_bbox in zip(label, label)
                ]  # [[batch_size, class_index, x_center, y_center, height, width], ...]

                if len(out_images) == batch_size:
                    out_images = np.array(out_images, dtype=np.float32)
                    out_labels = np.array(out_labels, dtype=np.float32)
                    # BGR转换成RGB
                    out_images = out_images[:, :, :, ::-1]
                    # 维度改成(n,c,h,w)
                    out_images = out_images.transpose(0, 3, 1, 2)
                    # 归一化 -1 ~ 1
                    out_images = out_images * 2 / 255 - 1
                    yield torch.from_numpy(out_images), torch.from_numpy(out_labels)
                    out_images = []
                    out_labels = []


def hpmp_data_generator(data_path, batch_size):
    """
    生成训练数据
    :param load_data: 加载数据function, 返回数据root_path
    :param image_list: 图片文件的数据集地址
    :param label_list: 标签文件的数据集地址
    :param batch_size: 每批取多少张图片
    :param resize: 输出的图片尺寸，(h,w)
    :param crop_offset: 将原始图片截掉多少，(start_height, start_width)
    :return: out_images=shape(batch_size,c,h,w),out_labels=shape(batch_size,h,w)
    """
    out_images = []
    out_labels = []

    print("The load data data path %s" % data_path)
    image_list = get_all_filename(data_path)
    check_path = image_list[0]
    if not os.path.isfile(check_path):
        raise Exception("Check that the data set path `{}` is correct".format(check_path))

    indices = np.arange(0, len(image_list))  # 索引

    all_standard_dict = get_all_image(r"D:\AI\project\aigame\outputs\all")

    while True:
        np.random.shuffle(indices)
        for i in indices:
            try:
                image = Image.open(image_list[i])  # (h,w,c) RGB格式
                label = get_image_hpmp(image_list[i],
                                       [43, 14, 138 + 1, 23 + 1],
                                       [43, 30, 138 + 1, 39 + 1],
                                       all_standard_dict)  # 2,2,digital_bits
            except:  # 发生异常，执行这块代码
                warnings.warn("The image file `{}` or label handle `{}` is error".format(image_list[i]))
            else:  # 如果没有异常执行这块代码
                if image is None:
                    warnings.warn("The image file `{}` is no exists".format(image_list[i]))
                    continue

                out_images.append(np.array(image))
                out_labels.append(label)

                if len(out_images) == batch_size:
                    out_images = np.array(out_images, dtype=np.float32)
                    out_labels = np.array(out_labels, dtype=np.int64)

                    # 维度改成(n,c,h,w)
                    out_images = out_images.transpose(0, 3, 1, 2)
                    # 归一化 -1 ~ 1
                    out_images = out_images * 2 / 255 - 1
                    yield torch.from_numpy(out_images), torch.from_numpy(out_labels).long()
                    out_images = []
                    out_labels = []


def _test_data_generator(image_list, batch_size, resize: Tuple,
                         crop_offset: Tuple = (0, 0)):
    """
    生成数据
    :param image_list: 图片路径列表, 必须是绝对路径
    :param batch_size: 每批取多少张图片
    :param resize: 输出的图片尺寸，(h,w)
    :param crop_offset: 将原始图片截掉多少，(start_height, start_width)
    :return: out_images=shape(batch_size,c,h,w),out_labels=shape(batch_size,h,w)
    """
    indices = np.arange(0, len(image_list))  # 索引
    original_images = []
    original_hw_size = []
    out_images = []
    out_images_filename = []

    check_path = image_list[0]
    if not os.path.isfile(check_path):
        raise Exception("Check that the data set path `{}` is correct".format(check_path))
    while True:
        np.random.shuffle(indices)
        for i in indices:
            try:
                image = cv2.imread(image_list[i])  # (h,w,c) BGR格式
            except:  # 发生异常，执行这块代码
                warnings.warn("The image file `{}` is no exists".format(image_list[i]))
            else:  # 如果没有异常执行这块代码
                if image is None:
                    warnings.warn("The image file `{}` is no exists".format(image_list[i]))
                    continue

                # 记录原始图与size
                original_images.append(image[:, :, ::-1])  # BGR转换成RGB
                original_hw_size.append(image.shape[0:2])

                if resize is not None and crop_offset is not None:
                    # crop & resize
                    image, label = crop_resize(image, None, resize, crop_offset)  # image=shape(h,w,c), label=shape(h,w)
                else:
                    print("The out_size and crop_offset is not set.")

                out_images.append(image)
                out_images_filename.append(image_list[i])
                if len(out_images) == batch_size:
                    out_images = np.array(out_images, dtype=np.float32)
                    # BGR转换成RGB
                    out_images = out_images[:, :, :, ::-1]
                    # 维度改成(n,c,h,w)
                    out_images = out_images.transpose(0, 3, 1, 2)
                    # 归一化 -1 ~ 1
                    out_images = out_images * 2 / 255 - 1
                    yield torch.from_numpy(out_images), original_images, original_hw_size
                    original_images = []
                    original_hw_size = []
                    out_images = []
                    out_images_filename = []


def test_data_generator(data_root_path, batch_size, resize: Tuple,
                        crop_offset: Tuple = (0, 0)):
    """
    生成数据
    :param data_root_path: 图片存放目录
    :param batch_size: 每批取多少张图片
    :param resize: 输出的图片尺寸，(h,w)
    :param crop_offset: 将原始图片截掉多少，(start_height, start_width)
    :return: out_images=shape(batch_size,c,h,w),out_labels=shape(batch_size,h,w)
    """
    image_list = get_all_filename(data_root_path)
    print("The load data root path %s" % data_root_path)

    return _test_data_generator(image_list, batch_size, resize, crop_offset), len(image_list)


def read_image(image_path, resize, crop_offset):
    ori_image = cv2.imread(image_path)  # (h,w,c) BGR格式
    image = ori_image
    ori_size = image.shape[0:2]
    if resize is not None and crop_offset is not None:
        # crop & resize
        image, label = crop_resize(image, None, resize, crop_offset)  # image=shape(h,w,c), label=shape(h,w)
    else:
        print("The out_size and crop_offset is not set.")

    # out_labels.append(label)
    # out_images_filename.append(image_list[i])
    # if len(out_images) == batch_size:
    out_images = np.array([image], dtype=np.float32)
    # out_labels = np.array(out_labels, dtype=np.int64)
    # BGR转换成RGB
    out_images = out_images[:, :, :, ::-1]
    # 维度改成(n,c,h,w)
    out_images = out_images.transpose(0, 3, 1, 2)
    # 归一化 -1 ~ 1
    out_images = out_images * 2 / 255 - 1
    return torch.from_numpy(out_images), ori_size, ori_image[:, :, ::-1]


def recover_image(image, resize, crop_offset):
    roi_image, roi_label = recover_crop_resize(image, None, resize, crop_offset)
    return roi_image


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
