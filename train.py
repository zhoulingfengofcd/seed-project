from utils.operation import *


if __name__ == '__main__':
    train(in_channels=3, out_channels=8, net_name="deeplab", lr=0.001, csv_path="./data_list/train1.csv",
          image_root=r"D:\AI\project\data\baidu_lane_line\original",
          label_root=r"D:\AI\project\data\baidu_lane_line\original",
          batch_size=4, out_size=(224, 224), crop_offset=(0, 0),
          epoch_begin=0, epoch_num=1,
          num_classes=8,
          net_state_path="./outputs/model"
          )
