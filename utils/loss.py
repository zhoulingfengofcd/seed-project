

# AABB(Axis-Aligned Bouding Box) iou
def compute_iou(predicts, labels, num_classes):
    """
    计算iou
    :param predicts: shape=(-1, classes)
    :param labels: shape=(-1, 1)
    :param num_classes: 分类数量
    :return:
    """