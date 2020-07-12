from utils.detection import *
from utils.file import *

if __name__ == '__main__':
    # train(in_channels=3, out_channels=8, net_name="deeplabv3", lr=0.001, csv_path="./data_list/train.csv",
    #       load_data=copy_dataset(copy_to_local_root=r"D:\AI\project\data\baidu_lane_line\original",
    #                              source_data_path=None),
    #       batch_size=2, resize=(512, 512), crop_offset=(730, 0),
    #       epoch_begin=0, epoch_num=1,
    #       num_classes=8,
    #       save_model=save_model(local_root="./outputs/model"),
    #       load_resnet_weight=load_weight(
    #           pretrained_weights_path=r"C:\Users\zlf\.cache\torch\checkpoints\resnet101-5d3b4d8f.pth",
    #           copy_to_local_root=None),
    #       )
    train_valid(in_channels=3, out_channels=8, net_name="yolov3", lr=0.001,
                # train_csv_path="./data_list/huawei_rubbish_train.csv",
                train_csv_path="./data_list/train3.csv",
                load_train_data=copy_dataset(copy_to_local_root=r"D:\AI\project\data\huawei_rubbish_classification",
                                             source_data_path=None),
                # valid_csv_path="./data_list/huawei_rubbish_valid.csv",
                valid_csv_path="./data_list/valid3.csv",
                load_valid_data=copy_dataset(copy_to_local_root=r"D:\AI\project\data\huawei_rubbish_classification",
                                             source_data_path=None),
                batch_size=3, resize=(416, 416), crop_offset=(0, 0),
                epoch_begin=0, epoch_num=1,
                num_classes=44,
                load_classes=read_lines(r"D:\AI\project\data\huawei_rubbish_classification\trainval\train_classes.txt"),
                # anchors=[(412, 406), (373, 265), (296, 372), (270, 159), (182, 309), (151, 93), (101, 195), (72, 76), (41, 34)],
                anchors=[(487, 490), (352, 467), (458, 355),  (434, 188), (291, 298), (189, 445),  (248, 142), (136, 226), (82, 84)],
                lr_strategy=[
                    [0.001],  # epoch 0
                    [0.001],  # epoch 1
                    [0.001],  # epoch 2
                    [0.001, 0.0006, 0.0003, 0.0001, 0.0004, 0.0008, 0.001],  # epoch 3
                    [0.001, 0.0006, 0.0003, 0.0001, 0.0004, 0.0008, 0.001],  # epoch 4
                    [0.001, 0.0006, 0.0003, 0.0001, 0.0004, 0.0008, 0.001],  # epoch 5
                    [0.0004, 0.0003, 0.0002, 0.0001, 0.0002, 0.0003, 0.0004],  # epoch 6
                    [0.00004, 0.00003, 0.00002, 0.00001, 0.00002, 0.00003, 0.00004],  # epoch 7
                    [0.00004, 0.00003, 0.00002, 0.00001, 0.00002, 0.00003, 0.00004],  # epoch 8
                    [0.00004, 0.00003, 0.00002, 0.00001, 0.00002, 0.00003, 0.00004],  # epoch 9
                ],
                save_model=save_model(local_root="./outputs/model"),
                loss_type=LossType.dice_loss,
                # load_resnet_weight=load_weight(
                #     pretrained_weights_path=r"C:\Users\zlf\.cache\torch\checkpoints\resnet101-5d3b4d8f.pth",
                #     copy_to_local_root=None),
                yolov3_model_json='./nets/yolo/yolov3-44.json',
                # load_state_dict=load_weight(
                #     pretrained_weights_path=r"D:\AI\project\data\weights\yolov3-44\ckpt_3_7432.67_0.00.pth",
                #     copy_to_local_root=None)
                )
