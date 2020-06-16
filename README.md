## 项目结构说明
```
——config 项目配置参数  
——data_list 划分好的数据集，如train.csv，valid.csv  
——nets 网络模型  
——outputs 输出文件  
  |——model 训练过程中保存的模型相关文件  
  |——log 输出日志  
——utils 各种工具  
——weights 预加载权重文件  
inference.py 测试脚本  
train.py 训练脚本  
```

## 模型定义说明
根据一个字典定义，创建模型，其设计是将网络作为一个有向图来定义，参数有：  
net：网络模型参数  
layers：每层具体定义，并作为图的节点  
adjacency：图节点的邻接表，网络的连接由邻接表指定  
start_to_end：输入到输出的列表，支持多输入、多输出  
layers内参数：请参考pytorch的对应方法参数  
  
示例：
```
{
    "net": {
        "in_channels": 3  # 必须
    },
    "layers": {
        "0": {
            "type": "con2d",  # 必须
            "out_channels": 3,  # 必须
            "kernel_size": 3,  # 必须
            "stride": 1,  # 默认
            "padding": 0,  # 默认
            "dilation": 1,  # 默认
            "groups": 1,  # 默认
            "bias": True,  # 默认
            "padding_mode": "zeros"  # 默认
        },
        "1": {
            "type": "maxpool2d",  # 必须
            "kernel_size": 3,  # 必须
            "stride": 3,  # 默认=kernel_size
            "padding": 0,  # 默认
            "dilation": 1,  # 默认
            "return_indices": False,  # 默认
            "ceil_mode": False  # 默认
        },
        "2": {
            "type": "interpolate",  # 必须
            # "size": (46, 46),  # size与scale_factor二选一, 输出image大小
            "scale_factor": 2,  # size与scale_factor二选一, 指定输出为输入的多少倍数
            "mode": "nearest",  # 默认
            "align_corners": None,  # 默认
            "recompute_scale_factor": None  # 默认
        },
        "3": {
            "type": "concat"  # 必须
        },
        "4": {
            "type": "shortcut"  # 必须
        },
        "5": {
            "type": "maxpool2d",  # 必须
            "kernel_size": 3,  # 必须
            "stride": 3,  # 默认=kernel_size
            "padding": 0,  # 默认
            "dilation": 1,  # 默认
            "return_indices": False,  # 默认
            "ceil_mode": False  # 默认
        },
        "6": {
            "type": "interpolate",  # 必须
            # "size": (46, 46),  # size与scale_factor二选一, 输出image大小
            "scale_factor": 2,  # size与scale_factor二选一, 指定输出为输入的多少倍数
            "mode": "nearest",  # 默认
            "align_corners": None,  # 默认
            "recompute_scale_factor": None  # 默认
        },
        "7": {
            "type": "concat"  # 必须
        },
        "8": {
            "type": "shortcut"  # 必须
        },
        "9": {
            "type": "shortcut"  # 必须
        }
    },
    "adjacency": {
        "0": ["1", "5"],
        "1": ["2"],
        "2": ["3"],
        "3": ["4"],

        "5": ["6"],
        "6": ["7"],
        "7": ["8"],
        "8": ["9"],
        "4": ["9"]
    },
    "start_to_end": [("0", "9")]
}
```