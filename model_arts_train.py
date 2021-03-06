from utils.operation import *
from utils.file import *
import argparse

if __name__ == '__main__':
    print("start model arts train.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', required=True, type=str, help='the training and validation data path')
    parser.add_argument('--train_url', required=True, type=str, help='the path to save training outputs')
    parser.add_argument('--init_method', default='', type=str, help='the training output results on local')
    opt = parser.parse_args()
    print(opt)
    # train(in_channels=3, out_channels=8, net_name="deeplabv3", lr=0.001, csv_path="seed-project/data_list/train.csv",
    #       load_data=copy_dataset(copy_to_local_root="/cache/datasets",
    #                              source_data_path='s3://zlf-rubbish-data/datasets/original.zip'),
    #       batch_size=16, resize=(512, 512), crop_offset=(730, 0),
    #       epoch_begin=0, epoch_num=5,
    #       num_classes=8,
    #       save_model=save_model(local_root="/cache/model", copy_root="s3://zlf-rubbish-data/outputs/"),
    #       load_resnet_weight=load_weight(
    #           pretrained_weights_path='s3://zlf-rubbish-data/weights/resnet101-5d3b4d8f.pth',
    #           copy_to_local_root='/cache/weight'),
    #       )
    train_valid(in_channels=3, out_channels=8, net_name="deeplabv3", lr=0.001,
                train_csv_path="seed-project/data_list/train.csv",
                load_train_data=copy_dataset(copy_to_local_root="/cache/datasets",
                                             source_data_path='s3://zlf-rubbish-data/datasets/original.zip'),
                valid_csv_path="seed-project/data_list/valid.csv",
                load_valid_data=copy_dataset(copy_to_local_root="/cache/datasets",
                                             source_data_path='s3://zlf-rubbish-data/datasets/original.zip'),
                batch_size=16, resize=(512, 512), crop_offset=(730, 0),
                epoch_begin=0, epoch_num=5,
                num_classes=8,
                save_model=save_model(local_root="/cache/model", copy_root="s3://zlf-rubbish-data/outputs/"),
                loss_type=LossType.dice_loss,
                load_resnet_weight=load_weight(
                    pretrained_weights_path='s3://zlf-rubbish-data/weights/resnet101-5d3b4d8f.pth',
                    copy_to_local_root='/cache/weight'),
                )
