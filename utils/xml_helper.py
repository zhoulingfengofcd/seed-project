# -*- coding=utf-8 -*-
import xml.etree.ElementTree as ET
import xml.dom.minidom as DOC
import os
import warnings


def parse_xml(xml_path, image_shape=None):
    """
    从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
    :param xml_path: xml的文件路径
    :return: 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
    """
    tree = ET.parse(xml_path)		
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    if image_shape is not None:
        if not h == image_shape[0] or not w == image_shape[1]:
            warnings.warn("Not equal in height or width {} {} {} {}".format(image_shape, h, w, xml_path))
            # raise ValueError("Not equal in width or height", image_shape, h, w, xml_path)

    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(box.find('xmin').text)
        y_min = int(box.find('ymin').text)
        x_max = int(box.find('xmax').text)
        y_max = int(box.find('ymax').text)
        coords.append([x_min, y_min, x_max, y_max, name])
        if x_max > w or y_max > h:
            raise ValueError('Image with annotation error x_max > w or y_max > h', x_max, w, y_max, h, xml_path)
        if x_min < 0 or y_min < 0:
            raise ValueError('Image with annotation error x_min < 0 or y_min < 0', x_min, y_min, xml_path)
    return coords


def convert_xyxy2xyhw(image, bboxes):
    """

    :param image: (h,w)
    :param bboxes: [[x_min, y_min, x_max, y_max], ...]
    :return:
    """
    convert_list = []
    h = image[0]
    w = image[1]
    for box in bboxes:
        width = round((box[2] - box[0]) / w, 6)
        height = round((box[3] - box[1]) / h, 6)
        x_center = round(((box[2] + box[0]) / 2) / w, 6)
        y_center = round(((box[3] + box[1]) / 2) / h, 6)

        if box[2] > w+1 or box[3] > h+1:
            raise Exception('Image with annotation error x_max > w or y_max > h', box[2], w, box[3], h)
        if box[0] < -1 or box[1] < -1:
            raise Exception('Image with annotation error x_min < 0 or y_min < 0', box[0], box[1])
        convert_list.append([x_center, y_center, height, width])
    return convert_list


def parse_convert_xml(xml_path):
    """
        从xml文件中提取bounding box信息, 格式为[[x_center, y_center, height, width, name], ...]
        :param xml_path: xml的文件路径
        :return: 从xml文件中提取bounding box信息, 格式为[[x_center, y_center, height, width, name], ...]
        """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        xmlbox = obj.find('bndbox')

        box = [int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
               int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)]
        width = round((box[2] - box[0]) / w, 6)
        height = round((box[3] - box[1]) / h, 6)
        x_center = round(((box[2] + box[0]) / 2) / w, 6)
        y_center = round(((box[3] + box[1]) / 2) / h, 6)

        if box[2] > w or box[3] > h:
            raise Exception('Image with annotation error:', xml_path)
        if box[0] < 0 or box[1] < 0:
            raise Exception('Image with annotation error:', xml_path)

        coords.append([x_center, y_center, height, width, name])
    return coords


def generate_xml(img_name, coords, img_size, out_root_path):
    """
    将bounding box信息写入xml文件中, bouding box格式为[[x_min, y_min, x_max, y_max, name]]
    :param img_name: 图片名称，如a.jpg
    :param coords: 坐标list，格式为[[x_min, y_min, x_max, y_max, name]]，name为概况的标注
    :param img_size: 图像的大小,格式为[h,w,c]
    :param out_root_path: xml文件输出的根路径
    :return:
    """
    doc = DOC.Document()  # 创建DOM文档对象

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    title = doc.createElement('folder')
    title_text = doc.createTextNode('Tianchi')
    title.appendChild(title_text)
    annotation.appendChild(title)

    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)

    source = doc.createElement('source')
    annotation.appendChild(source)

    title = doc.createElement('database')
    title_text = doc.createTextNode('The Tianchi Database')
    title.appendChild(title_text)
    source.appendChild(title)

    title = doc.createElement('annotation')
    title_text = doc.createTextNode('Tianchi')
    title.appendChild(title_text)
    source.appendChild(title)

    size = doc.createElement('size')
    annotation.appendChild(size)

    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)

    for coord in coords:

        object = doc.createElement('object')
        annotation.appendChild(object)

        title = doc.createElement('name')
        title_text = doc.createTextNode(coord[4])
        title.appendChild(title_text)
        object.appendChild(title)

        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        object.appendChild(difficult)

        bndbox = doc.createElement('bndbox')
        object.appendChild(bndbox)
        title = doc.createElement('xmin')
        title_text = doc.createTextNode(str(int(float(coord[0]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymin')
        title_text = doc.createTextNode(str(int(float(coord[1]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('xmax')
        title_text = doc.createTextNode(str(int(float(coord[2]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymax')
        title_text = doc.createTextNode(str(int(float(coord[3]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)

    # 将DOM对象doc写入文件
    f = open(os.path.join(out_root_path, img_name[:-4]+'.xml'), 'w', encoding='utf-8')
    f.write(doc.toprettyxml(indent=''))
    f.close()
