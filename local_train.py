from utils.operation import *
from utils.file import *

if __name__ == '__main__':
    train(in_channels=3, out_channels=8, net_name="deeplabv3", lr=0.001, csv_path="./data_list/train1.csv",
          load_data=copy_dataset(copy_to_local_root=r"D:\AI\project\data\baidu_lane_line\original",
                                 source_data_path=None),
          batch_size=4, out_size=(224, 224), crop_offset=(0, 0),
          epoch_begin=0, epoch_num=1,
          num_classes=8,
          save_model=save_model(local_root="./outputs/model"),
          load_weight=load_weight(
                pretrained_weights_path=r"C:\Users\zlf\.cache\torch\checkpoints\resnet101-5d3b4d8f.pth",
                copy_to_local_root=None),
          )