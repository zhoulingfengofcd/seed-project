from utils.operation import *
from utils.file import *
import argparse

if __name__ == '__main__':
    print("start model arts train.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', required=False, type=str, help='the training and validation data path')
    parser.add_argument('--train_url', required=False, type=str, help='the path to save training outputs')
    parser.add_argument('--init_method', default='', type=str, help='the training output results on local')
    opt = parser.parse_args()
    print(opt)
    test(
        in_channels=3, out_channels=8, net_name="deeplabv3",
        weights_path=r"D:\AI\project\data\weights\ckpt_0_50.18.pth",
        image_path=r"D:\AI\project\data\baidu_lane_line\original\Road03/ColorImage_road03/ColorImage/Record003/Camera 5/171206_030116718_Camera_5.jpg", out_size=None, crop_offset=None, pretrained=False
    )

