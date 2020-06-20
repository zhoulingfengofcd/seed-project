import os
import shutil
import torch
import zipfile
try:
    import moxing as mox
    mox.file.shift('os', 'mox')
except:
    print('not use moxing')


def load_weight(pretrained_weights_path, copy_to_local_root=None):
    """
    1、只传pretrained_weights_path参数，则直接从该路径加载预训练权重文件
    2、pretrained_weights_path, copy_to_local_root两个参数都传了，则会从pretrained_weights_path路径拷贝到
    copy_to_local_root/weight目录下,再加载权重文件
    :param pretrained_weights_path: 预训练权重文件的绝对路径
    :param copy_to_local_root: 将预训练权重文件拷贝到这个指定的目录加载
    :return: 加载好的权重文件
    """
    def _load_weight():
        if pretrained_weights_path is None:
            raise Exception("The `pretrained_weights_path` parameter cannot be None")

        if copy_to_local_root is not None:  # 先将pretrained_weights_path文件拷贝到copy_to_local_root/weight目录下,再加载
            _, weights_name = os.path.split(pretrained_weights_path)
            local_weights_path = os.path.join(copy_to_local_root, 'weight/' + weights_name)
            if not os.path.isfile(local_weights_path):
                shutil.copyfile(pretrained_weights_path, local_weights_path)
                print("copy file `{}` is {}.".format(local_weights_path, os.path.isfile(local_weights_path)))
            else:
                print("local_weights_path %s is already exist, skip copy" % local_weights_path)
            return torch.load(local_weights_path)
        else:  # 直接加载预训练权重
            return torch.load(pretrained_weights_path)

    return _load_weight


def _get_dir(dir_path):
    dir_files = os.listdir(dir_path)  # 得到该文件夹下所有的文件
    dirs = []
    for file in dir_files:
        file_path = os.path.join(dir_path, file)  # 路径拼接成绝对路径
        if os.path.isdir(file_path):  # 如果目录，就递归子目录
            dirs.append(file_path)
    return dirs


def print_dir(root_path):
    dir_files = os.listdir(root_path)  # 得到该文件夹下所有的文件
    for file in dir_files:
        file_path = os.path.join(root_path, file)  # 路径拼接成绝对路径
        if os.path.isdir(file_path):  # 如果目录，就递归子目录
            print(file_path)
            print_dir(file_path)


def copy_dataset(copy_to_local_root, source_data_path=None):
    """
    拷贝数据
    :param copy_to_local_root: 数据拷贝目录, 如果目录已存在, 则将不会拷贝, 否则将拷贝, 如果源数据是zip文件, 拷贝后将解压, 返回解压后的目录
    :param source_data_path: 源数据的完整路径(需包含文件名)
    :return:
    """
    def _copy_dataset():
        # local_data_root = os.path.join(copy_to_local_root, 'datasets')
        local_data_root = copy_to_local_root
        if not os.path.exists(local_data_root):
            filename = os.path.basename(source_data_path)
            # 从source_data_path路径拷贝数据集到copy_to_local_root目录下
            dst = os.path.join(local_data_root, filename)
            if not os.path.isfile(dst):
                print("start copy data from `{}` to `{}` ...".format(source_data_path, dst))
                shutil.copyfile(source_data_path, dst)
                print('copy file {} to {} is {}!'.format(source_data_path, dst, os.path.isfile(dst)))
            else:
                print("dst %s is already exist, skip copy" % dst)

            dirs = _get_dir(local_data_root)
            if len(dirs) == 1:
                print("return data dir %s" % dirs[0])
                return dirs[0]
            elif len(dirs) > 1:
                raise Exception("Multiple folders already exist under this directory %s" % local_data_root)
            else:
                cached_file = dst
                # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
                #       We deliberately don't handle tarfile here since our legacy serialization format was in tar.
                #       E.g. resnet18-5c106cde.pth which is widely used.
                if zipfile.is_zipfile(cached_file):
                    with zipfile.ZipFile(cached_file) as cached_zipfile:
                        members = cached_zipfile.infolist()
                        if not (members[0].filename.count("/") == 1 and members[1].filename.count("/") > 1):
                            raise RuntimeError('Only one file(not dir) is allowed in the zipfile the first menber.')
                        cached_zipfile.extractall(local_data_root)
                        extraced_name = members[0].filename
                        cached_file = os.path.join(local_data_root, extraced_name)
                        print("unzip success, cached_file is %s" % cached_file)
                        return cached_file
                else:
                    raise Exception("The cached_file `{}` is not zip file.".format(cached_file))
        else:
            print('local_data_root %s is already exist, skip copy' % local_data_root)
            dirs = _get_dir(local_data_root)
            if len(dirs) == 1:
                print("return data dir %s" % dirs[0])
                return dirs[0]
            else:
                print("return data dir %s" % local_data_root)
                return local_data_root
    return _copy_dataset


def save_model(local_root, copy_root=None):
    """
    保存模型
    :param local_root: 本地保存的目录
    :param copy_root: 如果需要拷贝到其他目录，可以传此参数
    :return:
    """
    def _save_model(model, temp_model_name):
        if not os.path.isdir(local_root):
            os.makedirs(local_root)
        ckpt_name = os.path.join(local_root, temp_model_name)
        torch.save(model.state_dict(), ckpt_name)
        if copy_root is not None:
            shutil.copytree(ckpt_name, os.path.join(copy_root, temp_model_name))
            print("copy the model {} to {}".format(ckpt_name, os.path.join(copy_root, temp_model_name)))
        else:
            print("only save model in %s" % ckpt_name)
    return _save_model


if __name__ == '__main__':
    copy_dataset(r"D:\\", r"D:\AI\project\data\baidu_lane_line\original")