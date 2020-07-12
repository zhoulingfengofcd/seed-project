"""
自定义模型创建工具
完善中...
"""
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from utils.graph import search_network_path
import torch
import warnings
from enum import Enum
from utils.file import read_json


class LayerType(Enum):
    Conv2d = "Conv2d"
    MaxPool2d = "MaxPool2d"
    interpolate = "interpolate"
    concat = "concat"
    shortcut = "shortcut"
    ReLU = "ReLU"
    BatchNorm2d = "BatchNorm2d"
    AdaptiveAvgPool2d = "AdaptiveAvgPool2d"
    Linear = "Linear"
    LeakyReLU = "LeakyReLU"


def create_layer(layer_def: dict, in_channel_list: list):
    if layer_def is None or len(layer_def) == 0:
        raise Exception("The layer_def parameter can not be None")
    layer_keys = layer_def.keys()
    if "type" not in layer_keys:
        raise Exception("The type parameter can not be None")
    layer_type = layer_def["type"]
    in_channels = sum(in_channel_list)

    # 二维卷积层
    if layer_type == LayerType.Conv2d.value:
        # 卷积层，无需定义所有参数，只需定义out_channels、kernel_size，其他参数都有默认值
        if 'out_channels' not in layer_keys:
            raise Exception("The out_channels parameter can not be None")
        if 'kernel_size' not in layer_keys:
            raise Exception("The kernel_size parameter can not be None")

        out_channels = layer_def['out_channels']
        layer_data = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=layer_def['kernel_size'],
            stride=1 if 'stride' not in layer_keys else layer_def['stride'],
            padding=0 if 'padding' not in layer_keys else layer_def['padding'],
            dilation=1 if 'dilation' not in layer_keys else layer_def['dilation'],
            groups=1 if 'groups' not in layer_keys else layer_def['groups'],
            bias=True if 'bias' not in layer_keys else layer_def['bias'],
            padding_mode='zeros' if 'padding_mode' not in layer_keys else layer_def['padding_mode']
        )
    elif layer_type == LayerType.MaxPool2d.value:
        if 'kernel_size' not in layer_keys:
            raise Exception("The kernel_size parameter can not be None")
        out_channels = in_channels
        layer_data = nn.MaxPool2d(
            kernel_size=layer_def['kernel_size'],
            stride=None if 'stride' not in layer_keys else layer_def['stride'],  # 如果未定义，默认kernel_size
            padding=0 if 'padding' not in layer_keys else layer_def['padding'],
            dilation=1 if 'dilation' not in layer_keys else layer_def['dilation'],
            return_indices=False if 'return_indices' not in layer_keys else layer_def['return_indices'],
            ceil_mode=False if 'ceil_mode' not in layer_keys else layer_def['ceil_mode']
        )
    elif layer_type == LayerType.interpolate.value:
        if "size" not in layer_keys and "scale_factor" not in layer_keys:
            raise Exception("The size and scale_factor parameter cannot be empty at the same time")
        elif "size" in layer_keys and "scale_factor" in layer_keys:
            raise Exception("only one of size or scale_factor should be defined")
        out_channels = in_channels
        layer_data = Interpolate(
            size=None if "size" not in layer_keys else layer_def["size"],
            scale_factor=None if "scale_factor" not in layer_keys else layer_def["scale_factor"],
            mode="nearest" if "mode" not in layer_keys else layer_def["mode"],
            align_corners=None if "align_corners" not in layer_keys else layer_def["align_corners"],
            recompute_scale_factor=None if "recompute_scale_factor" not in layer_keys else layer_def[
                "recompute_scale_factor"]
        )
    elif layer_type == LayerType.concat.value:
        out_channels = in_channels
        layer_data = EmptyLayer()  # 在执行时，再合并
    elif layer_type == LayerType.shortcut.value:
        out_channels = in_channel_list[0]
        layer_data = EmptyLayer()  # 在执行时，再相加
    elif layer_type == LayerType.ReLU.value:
        out_channels = in_channels
        layer_data = nn.ReLU(inplace=False if 'inplace' not in layer_keys else layer_def['inplace'])
    elif layer_type == LayerType.BatchNorm2d.value:
        out_channels = in_channels
        layer_data = nn.BatchNorm2d(
            num_features=in_channels if 'num_features' not in layer_keys else layer_def['num_features'],
            eps=1e-5 if 'eps' not in layer_keys else layer_def['eps'],
            momentum=0.1 if 'momentum' not in layer_keys else layer_def['momentum'],
            affine=True if 'affine' not in layer_keys else layer_def['affine'],
            track_running_stats=True if 'track_running_stats' not in layer_keys else layer_def['track_running_stats']
        )
    elif layer_type == LayerType.AdaptiveAvgPool2d.value:
        if 'output_size' not in layer_keys:
            raise Exception("The output_size parameter can not be None")
        out_channels = in_channels
        layer_data = nn.AdaptiveAvgPool2d(output_size=layer_def["output_size"])
    elif layer_type == LayerType.Linear.value:
        if 'in_features' not in layer_keys:
            raise Exception("The in_features parameter can not be None")
        if 'out_features' not in layer_keys:
            raise Exception("The out_features parameter can not be None")
        out_channels = layer_def["out_features"]
        layer_data = nn.Linear(
            in_features=layer_def["in_features"],
            out_features=layer_def["out_features"],
            bias=True if 'bias' not in layer_keys else layer_def['bias']
        )
    elif layer_type == LayerType.LeakyReLU.value:
        out_channels = in_channels
        layer_data = nn.LeakyReLU(
            negative_slope=1e-2 if 'negative_slope' not in layer_keys else layer_def['negative_slope'],
            inplace=False if 'inplace' not in layer_keys else layer_def['inplace']
        )
    else:
        raise Exception("The type `{}` undefined!".format(layer_def["type"]))
    return layer_data, out_channels, layer_def["type"]


class Interpolate(nn.Module):
    def __init__(self,
                 size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None
                 ):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, x):
        if self.recompute_scale_factor is None:  # 兼容1.3.0
            x = F.interpolate(input=x,
                              size=self.size,
                              scale_factor=self.scale_factor,
                              mode=self.mode,
                              align_corners=self.align_corners
                              )
        else:
            x = F.interpolate(input=x,
                              size=self.size,
                              scale_factor=self.scale_factor,
                              mode=self.mode,
                              align_corners=self.align_corners,
                              recompute_scale_factor=self.recompute_scale_factor
                              )
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


def create_model(net: dict, layers: dict, adjacency: dict, start_and_end: dict):
    if adjacency is None or len(adjacency.keys()) == 0:
        raise Exception("The adjacency parameter cannot be None")
    if start_and_end is None or 'start' not in start_and_end.keys() or 'end' not in start_and_end.keys() \
            or len(start_and_end['start']) == 0 or len(start_and_end['end']) == 0:
        raise Exception("The start_and_end parameter cannot be None")
    if net is None or len(net.keys()) == 0:
        raise Exception("The start_and_end parameter cannot be None")
    if "in_channels" not in net.keys():
        raise Exception("You need to define the `in_channels` parameter in net dictionary")

    output_node = start_and_end['end']
    start_node = start_and_end['start']

    # 邻接表转逆邻接表
    inverse_adjacency = dict()
    for adj_key in adjacency.keys():
        adj_list = adjacency[adj_key]
        if adj_list is None or len(adj_list) == 0:
            raise Exception("This adjacent node `{}` has no data".format(adj_key))
        for inv_key in adj_list:
            if inv_key in inverse_adjacency.keys():  # key是否已在逆邻接表
                if adj_key not in inverse_adjacency[inv_key]:  # value是否已在逆邻接表对应key的list中
                    inverse_adjacency[inv_key].append(adj_key)
            else:
                inverse_adjacency[inv_key] = [adj_key]

    layer_dict = OrderedDict()
    out_channels_dict = dict()

    path_list = search_network_path(start_node, output_node, adjacency, inverse_adjacency)
    print("计算节点数量={}".format(len(path_list)))
    if path_list is None or len(path_list) == 0:
        raise Exception("The start `{}` to end `{}` node path does not exist".format(start_node, output_node))

    for index, node in enumerate(path_list):  # 遍历所有节点
        if index == 0 and node not in start_node:
            raise Exception("The search path Exception!")

        if node not in start_node and [False for i in inverse_adjacency[node] if i not in layer_dict.keys()]:
            # 路径依赖的节点，如果有节点还未创建，抛出异常
            raise Exception("Incorrect node calculation sequence")
        if node in layer_dict.keys():
            raise Exception("Node to repeat")  # 该节点已经创建
        else:
            try:
                if node in start_node:
                    layer_define, out_channels, layer_type = create_layer(layers[node], [net["in_channels"]])
                else:
                    layer_define, out_channels, layer_type = create_layer(layers[node],
                                                                          [out_channels_dict[i] for i in
                                                                           inverse_adjacency[node]])
            except BaseException as e:
                raise Exception("node `{}`".format(node) + str(e.args))

            # 判断该节点是否被依赖
            is_depended = False
            if node in adjacency.keys():
                # 指向多个节点
                if len(adjacency[node]) > 1:
                    is_depended = True
                # 指向的节点，依赖多个节点
                for adjacency_node in adjacency[node]:
                    if adjacency_node in inverse_adjacency.keys():
                        if len(inverse_adjacency[adjacency_node]) > 1:
                            is_depended = True

            layer_dict[node] = ModelLayer(define=layer_define, layer_type=layer_type,
                                          inverse_adjacency=None if node not in inverse_adjacency.keys() else
                                          inverse_adjacency[node],
                                          adjacency=None if node not in adjacency.keys() else adjacency[node],
                                          is_depended=is_depended
                                          )

            out_channels_dict[node] = out_channels

    return layer_dict, start_node, output_node


class ModelLayer(nn.Module):
    def __init__(self, define, layer_type, inverse_adjacency, adjacency, is_depended):
        super(ModelLayer, self).__init__()
        self.define = define
        self.type = layer_type
        self.inverse_adjacency = inverse_adjacency
        self.adjacency = adjacency
        self.is_depended = is_depended


def get_depend(inverse_adjacency: list, output_cache: dict, last_layer_output: Tuple):
    if len(inverse_adjacency) == 1:
        depend_key = inverse_adjacency[0]
        if depend_key in output_cache.keys():
            depend = output_cache[depend_key]
        elif depend_key == last_layer_output[0]:
            depend = last_layer_output[1]
        else:
            raise Exception("This program logic exception")
    elif len(inverse_adjacency) > 1:
        depend = []
        for i in inverse_adjacency:
            if i in output_cache.keys():
                depend.append(output_cache[i])
            elif i == last_layer_output[0]:
                depend.append(last_layer_output[1])
            else:
                raise Exception("This program logic exception")
    return depend


class Model(nn.Module):
    def __init__(self, model_defs: dict):
        super(Model, self).__init__()
        if "net" not in model_defs.keys() or "layers" not in model_defs.keys() or "adjacency" not in model_defs.keys() \
                or "start_and_end" not in model_defs.keys():
            raise Exception("The model_defs need to define the `net` `layers` `adjacency` `start_and_end` parameter")

        net = model_defs["net"]
        layers = model_defs["layers"]
        adjacency = model_defs["adjacency"]
        start_and_end = model_defs["start_and_end"]
        layer_dict, self.start_node, self.output_node = create_model(
            net=net, layers=layers, adjacency=adjacency, start_and_end=start_and_end
        )
        self.model_keys = [k for k, v in layer_dict.items()]
        self.model_dict = nn.Sequential(
            OrderedDict([(str(k), v) for k, v in layer_dict.items()])  # 转换key为字符串，避免定义的key为整形报错
        )

    def forward(self, x):
        model_outputs = [0] * len(self.output_node)
        output_cache = dict()
        last_layer_output = ()
        for key, model_layer in zip(self.model_keys, self.model_dict):
            try:
                layer: ModelLayer = model_layer
                layer_type = layer.type
                layer_define = layer.define
                layer_inverse_adjacency = layer.inverse_adjacency
                layer_adjacency = layer.adjacency
                layer_is_depended = layer.is_depended

                if layer_type == LayerType.Conv2d.value:
                    if key in self.start_node:
                        result = layer_define(x)
                    else:
                        result = layer_define(get_depend(layer_inverse_adjacency, output_cache, last_layer_output))
                elif key in self.start_node:  # 网络开始的层，不能为后边的层
                    raise Exception("The start node is not convolution")
                elif layer_type == LayerType.MaxPool2d.value:
                    result = layer_define(get_depend(layer_inverse_adjacency, output_cache, last_layer_output))
                elif layer_type == LayerType.interpolate.value:
                    result = layer_define(get_depend(layer_inverse_adjacency, output_cache, last_layer_output))
                elif layer_type == LayerType.concat.value:
                    depend = get_depend(layer_inverse_adjacency, output_cache, last_layer_output)
                    if isinstance(depend, torch.Tensor):
                        warnings.warn("This concat is only one edges")
                        result = torch.cat([depend], 1)
                    else:
                        result = torch.cat(depend, 1)
                elif layer_type == LayerType.shortcut.value:
                    depend = get_depend(layer_inverse_adjacency, output_cache, last_layer_output)
                    if isinstance(depend, torch.Tensor):
                        warnings.warn("This shortcut `{}` is only one edges".format(key))
                        result = depend
                    else:
                        for i in range(1, len(depend)):
                            result = depend[i - 1] + depend[i]
                elif layer_type == LayerType.ReLU.value:
                    result = layer_define(get_depend(layer_inverse_adjacency, output_cache, last_layer_output))
                elif layer_type == LayerType.BatchNorm2d.value:
                    result = layer_define(get_depend(layer_inverse_adjacency, output_cache, last_layer_output))
                elif layer_type == LayerType.AdaptiveAvgPool2d.value:
                    result = layer_define(get_depend(layer_inverse_adjacency, output_cache, last_layer_output))
                elif layer_type == LayerType.Linear.value:
                    result = layer_define(get_depend(layer_inverse_adjacency, output_cache, last_layer_output))
                elif layer_type == LayerType.LeakyReLU.value:
                    result = layer_define(get_depend(layer_inverse_adjacency, output_cache, last_layer_output))
                else:
                    raise Exception("The type `{}` undefined!".format(layer_type))
                # 记录上一层
                last_layer_output = (key, result)
                # 记录依赖分支的层
                if layer_is_depended:
                    output_cache[key] = result
                # 记录输出层
                if key in self.output_node:
                    model_outputs[self.output_node.index(key)] = result
            except BaseException as e:
                raise Exception("node `{}`".format(key) + str(e.args))

        return model_outputs


def load_model_from_json1(path, in_channels):
    json = read_json(path)
    model_defs = {
        "net": {
            "in_channels": in_channels
        },
        "layers": {},
        "adjacency": {},
        "start_and_end": {}
    }

    # 遍历节点
    if 'nodes' not in json:
        raise Exception("No `nodes` are defined in the configuration")
    nodes = json['nodes']
    for node in nodes:
        # 节点类型
        if 'data' not in node or 'layer' not in node['data'] or 'type' not in node['data']['layer']:
            raise Exception("Node `type` must be defined")
        layer_type = node['data']['layer']['type']
        layer_def = {
            "type": layer_type
        }
        # 节点id
        if 'id' not in node:
            raise Exception("The node has no `id` defined")
        node_id = node['id']
        # 开始、结束节点
        if 'start_or_end' in node['data']['layer']:
            if node['data']['layer']['start_or_end'] == 'start':
                if 'start' in model_defs['start_and_end'].keys():
                    model_defs['start_and_end']['start'].append(node_id)
                else:
                    model_defs['start_and_end']['start'] = [node_id]
            elif node['data']['layer']['start_or_end'] == 'end':
                if 'end' in model_defs['start_and_end'].keys():
                    model_defs['start_and_end']['end'].append(node_id)
                else:
                    model_defs['start_and_end']['end'] = [node_id]
        # 节点参数
        if 'parameters' in node['data']:
            parameters = node['data']['parameters']
            for parameter in parameters:
                if 'label' not in parameter or 'type' not in parameter:
                    raise Exception("Parameter `label` or `type` must be defined")
                key = parameter['label']
                parameter_type = parameter['type']

                if 'value' in parameter and parameter['value'] is not None:
                    if parameter_type == 'TupleNumber':
                        layer_def[key] = tuple(parameter['value'])
                    else:
                        layer_def[key] = parameter['value']
                elif 'options' in parameter and 'initialValue' in parameter['options'] and parameter['options']['initialValue'] is not None:
                    if parameter_type == 'TupleNumber':
                        layer_def[key] = tuple(parameter['options']['initialValue'])
                    else:
                        layer_def[key] = parameter['options']['initialValue']

        model_defs['layers'][node_id] = layer_def

    if 'start' not in model_defs['start_and_end'].keys() or 'end' not in model_defs['start_and_end'].keys():
        raise Exception("No `start` or `end` node are defined in the configuration")

    # 遍历连线
    if 'lines' not in json:
        raise Exception("No `lines` are defined in the configuration")
    lines = json['lines']
    for line in lines:
        if 'id' not in line or 'from' not in line or 'id' not in line['from'] or 'to' not in line or 'id' not in line['to']:
            raise Exception("The node has no `id` or `from` or `to` defined")
        if line['from']['id'] not in model_defs['layers'].keys() or line['to']['id'] not in model_defs['layers'].keys():
            raise Exception("The node has no `id` defined in the nodes")

        if line['from']['id'] in model_defs['adjacency'].keys():
            model_defs['adjacency'][line['from']['id']].append(line['to']['id'])
        else:
            model_defs['adjacency'][line['from']['id']] = [line['to']['id']]

    print("配置文件节点数量={}".format(len(model_defs['layers'].keys())))

    model = Model(model_defs=model_defs)
    return model


def load_model_from_json2(path, in_channels):
    json = read_json(path)
    model_defs = {
        "net": {
            "in_channels": in_channels
        },
        "layers": {},
        "adjacency": {},
        "start_and_end": {}
    }

    # 遍历节点
    if 'pens' not in json:
        raise Exception("No `pens` are defined in the configuration")
    pens = json['pens']
    for node_or_line in pens:
        if 'type' not in node_or_line:
            raise Exception("Each entry in the pens must define the Type parameter")
        pens_type = node_or_line['type']
        if pens_type == 0:  # 节点
            node = node_or_line
            # 节点类型
            if 'data' not in node or 'layer' not in node['data'] or 'type' not in node['data']['layer']:
                raise Exception("Node `type` must be defined")
            layer_type = node['data']['layer']['type']
            layer_def = {
                "type": layer_type
            }
            # 节点id
            if 'id' not in node:
                raise Exception("The node has no `id` defined")
            node_id = node['id']
            # 开始、结束节点
            if 'start_or_end' in node['data']['layer']:
                if node['data']['layer']['start_or_end'] == 'start':
                    if 'start' in model_defs['start_and_end'].keys():
                        model_defs['start_and_end']['start'].append(node_id)
                    else:
                        model_defs['start_and_end']['start'] = [node_id]
                elif node['data']['layer']['start_or_end'] == 'end':
                    if 'end' in model_defs['start_and_end'].keys():
                        model_defs['start_and_end']['end'].append(node_id)
                    else:
                        model_defs['start_and_end']['end'] = [node_id]
            # 节点参数
            if 'parameters' in node['data']:
                parameters = node['data']['parameters']
                for parameter in parameters:
                    if 'label' not in parameter or 'type' not in parameter:
                        raise Exception("Parameter `label` or `type` must be defined")
                    key = parameter['label']
                    parameter_type = parameter['type']

                    if 'value' in parameter and parameter['value'] is not None:
                        if parameter_type == 'TupleNumber':
                            layer_def[key] = tuple(parameter['value'])
                        else:
                            layer_def[key] = parameter['value']
                    elif 'options' in parameter and 'initialValue' in parameter['options'] and parameter['options']['initialValue'] is not None:
                        if parameter_type == 'TupleNumber':
                            layer_def[key] = tuple(parameter['options']['initialValue'])
                        else:
                            layer_def[key] = parameter['options']['initialValue']

            model_defs['layers'][node_id] = layer_def

    # 遍历连线
    for node_or_line in pens:
        if 'type' not in node_or_line:
            raise Exception("Each entry in the pens must define the Type parameter")
        pens_type = node_or_line['type']
        if pens_type == 1:  # 连线
            line = node_or_line
            if 'id' not in line or 'from' not in line or 'id' not in line['from'] or 'to' not in line or 'id' not in \
                    line['to']:
                raise Exception("The line `{}` has no `id` or `from` or `to` defined".format(line))
            if line['from']['id'] not in model_defs['layers'].keys() or line['to']['id'] not in model_defs['layers'].keys():
                raise Exception("The line `{}` has no `id` defined in the lines".format(line))

            if line['from']['id'] in model_defs['adjacency'].keys():
                model_defs['adjacency'][line['from']['id']].append(line['to']['id'])
            else:
                model_defs['adjacency'][line['from']['id']] = [line['to']['id']]

    if 'start' not in model_defs['start_and_end'].keys() or 'end' not in model_defs['start_and_end'].keys():
        raise Exception("No `start` or `end` node are defined in the configuration")

    print("配置文件节点数量={}".format(len(model_defs['layers'].keys())))

    return Model(model_defs=model_defs)


if __name__ == '__main__':
    # _test_create_model()

    # model = torchvision.models.resnet101()
    # layers: dict = layer_convert_dict(model)
    # print(model)
    # print(layers)
    #
    # input = torch.randn(1, 3, 224, 224).requires_grad_(True)
    #
    # output, adjacency, start, end = get_data_cache(model, input)
    #
    # print(start, end)
    # print(adjacency)
    # graph = "digraph G{\n"
    # for key1 in adjacency.keys():
    #     for key2 in adjacency[key1]:
    #         graph += key1 + "->" + key2 + ";\n"
    # graph += "}\n"
    # print(graph)

    model = load_model_from_json1("yolo/yolov3-44.json", 3)
    print(model)
    input = torch.rand(1, 3, 416, 416)
    outputs = model(input)
    for output in outputs:
        print(output.shape)
    # from utils.file import save_model
    # save_model(r'D:\AI\project\data\weights')(model, "test1")

    # pth = r'D:\AI\project\data\weights\test1'
    # sta_dic = torch.load(pth)
    # print('.pth type:', type(sta_dic))
    # print('.pth len:', len(sta_dic))
    # print('--------------------------')
    # for k in sta_dic.keys():
    #     print(k, type(sta_dic[k]), sta_dic[k].shape)

