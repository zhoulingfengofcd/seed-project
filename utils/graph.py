import copy
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
# 如果图中汉字无法显示，请参照如下配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei']


def search(graph, start, end):
    '''
    从图数据graph中，搜索start->end的所有路径
    注意：返回的路径无交集，比如路径A、B都经过同一个点，那么只返回A或B其中一条路径
    :param graph: 图数据，格式如下：
    {
        '小张': ['小刘', '小王', '小红'],
        '小王': ['六六', '娇娇', '小曲'],
        '娇娇': ['宝宝', '花花', '喵喵'],
        '六六': ['小罗', '奥巴马']
    }
    :param start: 开始节点
    :param end: 结束节点
    :return: 所有路径列表
    '''
    check = [[start]]  # 搜索路径
    visited = set()  # 已访问过的节点
    finished = []
    while check:
        # 1、取一条未完成的搜索路径
        # 队列（广度优先搜索）
        # path = check.pop(0)  # 取队列头
        # 栈（深度优先搜索）
        path = check.pop(-1)  # 取栈顶

        # 2、遍历该路径的下一层
        node = path[-1]
        # 如果节点已访问，则抛弃（本条路径），防止死循环
        if node in visited:
            continue
        # 遍历下一层
        for item in graph.get(node, []):
            new_path = path + [item]

            if item == end:
                finished.append(new_path)  # 找到终点，则该路径结束搜索
            else:
                check.append(new_path)  # 否则将路径添加到搜索路径中
        visited.add(node)
    return finished


def search_all_may(graph, start, end):
    '''
    从图数据graph中，搜索start->end的所有路径
    注意：返回的路径有交集, 图不能有闭环，否则会死循环
    :param graph: 图数据，格式如下：
    {
        '小张': ['小刘', '小王', '小红'],
        '小王': ['六六', '娇娇', '小曲'],
        '娇娇': ['宝宝', '花花', '喵喵'],
        '六六': ['小罗', '奥巴马']
    }
    :param start: 开始节点
    :param end: 结束节点
    :return: 所有路径列表
    '''
    check = [[start]]  # 搜索路径
    # visited = set()  # 已访问过的节点
    finished = []
    while check:
        # 1、取一条未完成的搜索路径
        # 队列（广度优先搜索）
        # path = check.pop(0)  # 取队列头
        # 栈（深度优先搜索）
        path = check.pop(-1)  # 取栈顶

        # 2、遍历该路径的下一层
        node = path[-1]
        # 如果节点已访问，则抛弃（本条路径），防止死循环
        # if node in visited:
        #     continue
        # 遍历下一层
        for item in graph.get(node, []):
            new_path = path + [item]

            if item == end:
                finished.append(new_path)  # 找到终点，则该路径结束搜索
            else:
                check.append(new_path)  # 否则将路径添加到搜索路径中
        # visited.add(node)
    return finished


def search_network_path(starts: list, ends: list, adjacency: dict, inverse_adjacency: dict):
    check = []
    for start in starts:
        check.append([start])

    finished = []  # 最终保存的路径
    # visited = set()
    while check:
        path = check.pop(-1)

        node = path[-1]  # 取路径最后一个节点

        if node in ends:  # 是结束节点
            finish(finished, path)
        elif node not in start and len(inverse_adjacency[node]) > 1:  # 不是开始节点，且依赖多个节点
            is_visited = True
            for item in inverse_adjacency[node]:
                if item not in finished and item not in path:
                    is_visited = False
            if is_visited:
                for item in adjacency[node]:
                    if item not in finished:
                        check.append([item])  # 指向的节点，成新分支
                finish(finished, path)
            else:
                finish(finished, path[:-1])
        elif len(adjacency[node]) > 1:  # 指向多个节点
            for item in adjacency[node]:
                if item not in finished:
                    check.append([item])  # 指向的多个节点，成新分支
            finish(finished, path)
        elif len(adjacency[node]) == 1:  # 指向一个节点
            check.append(path + adjacency[node])  # 将指向的节点添加到该路径中
        else:
            raise Exception("Path to interrupt")
    print(finished)
    return finished


def finish(finished: list, path: list):
    for item in path:
        if item not in finished:
            finished.append(item)
        else:
            print("重复遍历节点", item)
            # raise Exception("exist value")


if __name__ == '__main__':
    data = {
            '小张': ['小刘', '小王', '小红'],
            '小王': ['六六', '娇娇', '小曲'],
            '娇娇': ['宝宝', '花花', '喵喵'],
            '六六': ['小罗', '奥巴马'],
            '花花': ['奥巴马'],
            '小红': ['奥巴马'],
        }
    paths = search(data, "小张", "奥巴马")
    print(paths)
    connect_graph = nx.Graph(data)
    nx.draw(connect_graph, with_labels=True, node_size=6, font_size=8)
    plt.show()