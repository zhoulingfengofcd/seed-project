import torch
import torch.nn.functional as F


def _bbox_wh_iou(wh1, wh2):
    """
    求中心点重合的box的iou, 注意本方法已经假设传入的box中心点是重合的
    支持1对1,1对n(某个box与其他n个box的iou值)，n对n(跟1对1类似，最后求得对应位置的box的iou)，不支持m对n
    :param wh1: shape=(-1, 2),最后一维为(w,h)
    :param wh2: shape=(-1, 2),最后一维为(w,h)
    :return: 输出纬度跟随输入-1的最大纬度，shape=(-1)，wh1的-1纬度可以不与wh2的-1纬度一致，如wh1.shape=(2), wh2.shape=(2,2), 则输出=(2)
    """
    # wh2 = wh2.t()
    w1, h1 = wh1[..., 0], wh1[..., 1]
    w2, h2 = wh2[..., 0], wh2[..., 1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def _bbox_iou(box1, box2, x1y1x2y2=True):
    """
    求bounding boxs的iou
    :param box1: shape = (-1,4), example: box1.shape=box2.shape=(4) or (1,4)  or (1,2,4) 只需保证最后一维为4
    :param box2: shape = (-1,4), example: box1.shape=box2.shape=(4) or (1,4)  or (1,2,4) 只需保证最后一维为4
    :param x1y1x2y2: True box (x1,y1,x2,y2), False, box (x,y,w,h)
    :return: 输出纬度跟随输入纬度,如输入(-1,4), 输出(-1), 注意要保证box1、box2前边-1的纬度一致
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def _fast_hist(label_true, label_pred, n_class):
    """
    计算混淆矩阵
    :param label_true: shape=(h,w), 表示真实的标记，这里是一个二维数组，也可以理解为一张灰度图，每个像素点对应于一个类别（用数字表示0,1,2,…,n_class）
    :param label_pred: shape=(h,w), 表示预测结果，格式同label_true
    :param n_class: 类别总数
    :return: 混淆矩阵, shape=(n_class, n_class)
    """
    mask = (label_true >= 0) & (label_true < n_class)
    hist = torch.bincount(
        n_class * label_true[mask] +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def create_iou_loss(predicts, labels, num_classes):
    """
    计算iou
    :param predicts: shape=(-1, classes)
    :param labels: shape=(-1, 1)
    :param num_classes: 分类数量
    :return:
    """


def get_miou(predicts: torch.Tensor, labels: torch.Tensor, num_classes):
    """
    根据混淆矩阵，求miou(其过程不可导)
    :param predicts: shape=(n,c,h,w)
    :param labels: shape=(n,h,w),  one hot format
    :param num_classes: int should equal to channels of predicts
    :return: loss, mean_iou
    """
    pred = predicts.argmax(dim=1)  # 将预测值转换成one hot形式, (n, h, w)

    total_miou = 0.0
    for lt, lp in zip(labels, pred):  # 遍历每个样本
        hist = _fast_hist(label_true=lt, label_pred=lp, n_class=num_classes).float()
        iou = hist.diag()/(hist.sum(0) + hist.sum(1) - hist.diag())
        total_miou += iou.mean()

    return total_miou


def create_iou_loss(predicts: torch.Tensor, labels: torch.Tensor, num_classes):
    """
    计算iou损失
    see: https://zhuanlan.zhihu.com/p/101773544
    :param predicts: shape=(n,c,h,w)
    :param labels: shape=(n,h,w)
    :param num_classes: 分类的类别数
    :return: 本批次的平均iou loss值
    """
    softmax_predict = torch.softmax(predicts, dim=1)
    labels_one_hot = make_one_hot(labels, num_classes)
    inter = softmax_predict * labels_one_hot
    iou = inter.sum(dim=(-2, -1)) / (softmax_predict + labels_one_hot - inter).sum(dim=(-2, -1))
    return 1-iou.mean()


def _dice_loss(predicts: torch.Tensor, labels: torch.Tensor):
    """
    计算dice loss，注意perdicts与labels的纬度要对应
    计算公式: 1 - (2|X交Y| +1) / (|X| + |Y| +1)
    see: https://zhuanlan.zhihu.com/p/86704421
    see: https://zhuanlan.zhihu.com/p/101773544
    :param predicts: shape = (n,h,w) or (n,c,h,w)
    :param labels: shape = (n,h,w) or (n,c,h,w)
    :return: loss shape = (n) or (n,c)
    """
    inter = (predicts * labels).sum(dim=(-2, -1))
    loss = 1 - (2*inter+1) / (predicts.sum(dim=(-2, -1)) + labels.sum(dim=(-2, -1)) + 1)
    return loss


def make_one_hot(labels: torch.Tensor, num_classes):
    """
    将分割标签图转为每个类别的one-hot图
    :param labels: shape=(n,h,w)
    :param num_classes: 分类的类别数
    :return: shape=(n,c,h,w)
    """
    return F.one_hot(labels, num_classes).permute(0, 3, 1, 2)


def create_dice_loss(predicts: torch.Tensor, labels: torch.Tensor, num_classes, weights=None):
    """
    求dice loss损失
    :param predicts: shape=(n,c,h,w)
    :param labels: shape=(n,h,w)
    :param num_classes: 分类的类别数
    :param weights: 每个类别损失权重, shape=(num_classes)
    :return: 本批次的平均dice loss值
    """
    # 求每个通道的dice_loss
    nc_loss = _dice_loss(torch.softmax(predicts, dim=1), make_one_hot(labels, num_classes))
    if weights is not None:
        nc_loss = nc_loss * weights
    return nc_loss.mean()


def create_ce_loss(predicts: torch.Tensor, labels: torch.Tensor, num_classes):
    """
    创建loss
    :param predicts: shape=(n,c,h,w)
    :param labels: shape=(n,h,w) or shape=(n,1,h,w)
    :param num_classes: int should equal to channels of predicts
    :return: loss, mean_iou
    """
    # 维度换位(n,h,w,c)
    predicts = predicts.permute((0, 2, 3, 1))
    # reshape to (-1, num_classes) 每个像素在每种分类上都有一个概率
    predicts = predicts.reshape((-1, num_classes))
    # bce with dice
    ce_loss = F.cross_entropy(input=predicts, target=labels.flatten())  # log(softmax(input)) -> NLLLoss(input, target)
    return ce_loss


if __name__ == '__main__':
    # a = torch.Tensor([
    #     [
    #         [1, 2, 3],
    #         [4, 5, 6],
    #         [7, 8, 9]
    #     ],
    #     [
    #         [1, 2, 3],
    #         [4, 5, 6],
    #         [7, 8, 9]
    #     ]
    # ])
    # b = torch.Tensor([
    #     [
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 1]
    #     ],
    #     [
    #         [0, 0, 0],
    #         [0, 0, 1],
    #         [0, 0, 0]
    #     ]
    # ])
    # print(a.shape)
    # # print(a.view(a.size(0), -1))
    # # print(a.sum(dim=(-2, -1)))
    # print((a * b).sum(dim=(-2, -1)))
    # print(create_dice_loss(a, b))
    #
    # c = torch.Tensor([[
    #     [
    #         [1, 2, 3],
    #         [4, 5, 6],
    #         [7, 8, 9]
    #     ],
    #     [
    #         [1, 2, 3],
    #         [4, 5, 6],
    #         [7, 8, 9]
    #     ]
    # ]])
    # d = torch.Tensor([[
    #     [
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 1]
    #     ],
    #     [
    #         [0, 0, 0],
    #         [0, 0, 1],
    #         [0, 0, 0]
    #     ]
    # ]])
    # print(create_dice_loss(c, d))
    #
    # e = torch.Tensor(
    #     [1, 2, 3, 4]
    # )
    # f = torch.Tensor(
    #     [1, 2, 3, 4])
    # print(e.shape, f.shape)
    # print(_bbox_iou(e, f))
    #
    # g = torch.Tensor([[1, 2], [3, 4]])
    # h = torch.Tensor([[5, 6], [7, 8]])
    # print(_bbox_wh_iou(g, h))
    a = torch.Tensor([
        [
            [1, 2, 3],
            [4, 5, 6]
        ],
        [
            [1, 2, 3],
            [4, 5, 6]
        ]
    ])
    a.expand(2, 8, 2, 3)
    print(a)