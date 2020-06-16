import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
# 如果图中汉字无法显示，请参照如下配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei']


def search(graph, start, end):
    '''
    从图数据graph中，搜索start->end的所有路径
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