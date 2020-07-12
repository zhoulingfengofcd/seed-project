import torch
import numpy as np
import tqdm
from utils.loss import bbox_wh_iou, bbox_iou
# from terminaltables import AsciiTable


def to_cpu(tensor):
    return tensor.detach().cpu()


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyhw2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 3] / 2
    y[..., 1] = x[..., 1] - x[..., 2] / 2
    y[..., 2] = x[..., 0] + x[..., 3] / 2
    y[..., 3] = x[..., 1] + x[..., 2] / 2
    return y


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    非最大值抑制
    :param prediction: shape=(n,num_anchor*(f1_w*f1_h+f2_w*f2_h+f3_w*f3_h),(x_center,y_center,h,w,置信度,num_classes))
    :param conf_thres:
    :param nms_thres:
    :return: shape=(num_pred_box, (x1,y1,x2,y2,置信度,类别概率,类别))
    """
    """
    非最大值抑制
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xyhw2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)  # (num_pred, (x1,y1,x2,y2,置信度,类别概率,类别))
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def get_batch_statistics(outputs, targets, iou_threshold):
    """ outputs=list(n) list item shape=[num_pred_box,7], 其中7=(x1,y1,x2,y2,置信度,类别概率,类别)
    计算每个样本的正确率、预测得分和预测标签 targets=[[batch_size, class_index, x1, y1, x2, y2], ...]
    Compute true positives, predicted scores and predicted labels per sample
    """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]  # shape=(num_pred_box,4)
        pred_scores = output[:, 4]  # shape=(num_pred_box)
        pred_labels = output[:, -1]  # shape=(num_pred_box)

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]  # annotations=[[class_index, x1, y1, x2, y2],...]
        target_labels = annotations[:, 0] if len(annotations) else []  # target_labels=[class_index, ...]
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]  # target_boxes=[[x1, y1, x2, y2],...]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def get_yolo_output(predicts, num_classes, input_size, anchors, cuda=True,
                    labels=None, ignore_threshold=0.5,
                    obj_scale=None, noobj_scale=None, coord_scale=None, cls_scale=None):
    loss = 0
    yolo_outputs = []
    num_anchor = len(anchors) // len(predicts)
    metrics_table = []
    for index, pred in enumerate(predicts):
        output, layer_loss, metrics = get_yolo_layer_ouput(pred, num_classes, input_size,
                                                           anchors[index*num_anchor:(index+1)*num_anchor], cuda, labels,
                                                           ignore_threshold, obj_scale, noobj_scale, coord_scale, cls_scale)
        loss += layer_loss
        yolo_outputs.append(output)
        if metrics is not None and len(metrics.keys()) > 0:
            metrics_table.append(metrics)
    yolo_outputs = torch.cat(yolo_outputs, 1)
    # if len(metrics_table) > 0:
    #     print_metrics(metrics_table)
    return yolo_outputs, loss, metrics_table


def print_metrics(metrics_table):

    table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(metrics_table))]]]
    column_max_length = [len(string) for string in table[0]]
    for metrics_key in metrics_table[0].keys():
        row_metrics = [metrics_item[metrics_key] for metrics_item in metrics_table]
        table += [[metrics_key, *row_metrics]]
        column_max_length = [len(str(row_string)) if len(str(row_string)) > max_length else max_length for row_string, max_length in zip(table[-1], column_max_length)]

    for index, row in enumerate(table):
        if index == 0 or index == 1:
            for value_index, length in enumerate(column_max_length):
                if value_index == 0:
                    print('-' + ('-' * length), end='+')
                else:
                    print('-' * length, end='+')
            print()
        for column_index, (column, length) in enumerate(zip(row, column_max_length)):
            pad = ' ' * (length-len(str(column)))
            if column_index == 0:
                print('|' + str(column) + pad, end='|')
            else:
                print(str(column)+pad, end='|')
        print()
        if index == len(table)-1:
            for value_index, length in enumerate(column_max_length):
                if value_index == 0:
                    print('-' + ('-' * length), end='+')
                else:
                    print('-' * length, end='+')
            print()


def get_yolo_layer_ouput(predicts, num_classes, input_size, anchors, cuda=True,
                         labels=None, ignore_threshold=0.5,
                         obj_scale=None, noobj_scale=None, coord_scale=None, cls_scale=None):
    """

    :param predicts: shape=(n,c,h,w), 其中c=num_anchor*(5+num_classes)
    :param labels: shape=(box_num,6), 其中6=(n,class_index,box_x_center,box_y_center,box_h,box_w)
    :param num_classes: 分类类别数
    :param input_size: example: [h,w]
    :param anchors: 预选框高宽[(h1,w1), (h2, w2), ......], 其size=num_anchor, 示例[(41,34), (72,76), (101,195), (151,93), (182,309), (270,159), (296,372), (373,265), (412,406)]
    :param cuda:
    :return:
    """
    num_samples = predicts.size(0)  # n
    grid_h_size = predicts.size(2)  # h
    grid_w_size = predicts.size(3)  # w
    prediction = (
        # 纬度变换shape=(n,num_anchor,5+num_classes,grid_h_size,grid_w_size) (1,3,49,13,13)
        predicts.view(num_samples, len(anchors), 5 + num_classes, grid_h_size, grid_w_size)
            .permute(0, 1, 3, 4, 2)  # 纬度变换shape=(n,num_anchor,grid_h_size,grid_w_size,5+num_classes) (1,3,13,13,49)
            .contiguous()  # 纬度变换后，把tensor变成在内存中连续分布的形式
    )  # pred shape=(n,num_anchor,grid_h_size,grid_w_size,5+num_classes) (1,3,13,13,49)，batch_size=1，预选框数量=3，特征图大小=13*13,最后一维=(x,y,w,h)+1置信度+44分类

    x = torch.sigmoid(prediction[..., 0])  # Center x,shape=(1,3,13,13)
    y = torch.sigmoid(prediction[..., 1])  # Center y,shape=(1,3,13,13)
    h = prediction[..., 2]  # Height,shape=(1,3,13,13)
    w = prediction[..., 3]  # Width,shape=(1,3,13,13)
    pred_conf = torch.sigmoid(prediction[..., 4])  # Conf pred_conf.size(1,3,13,13),bbox的置信度，或者单元格内有没有物体的概率
    pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred. pred_cls.size=(1,3,13,13,44),每个类别的概率

    float_tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    grid_x = torch.arange(grid_w_size).repeat(grid_h_size, 1).view([1, 1, grid_h_size, grid_w_size]).type(
        float_tensor)  # size(1,1,13,13)
    grid_y = torch.arange(grid_h_size).repeat(grid_w_size, 1).t().view([1, 1, grid_h_size, grid_w_size]).type(
        float_tensor)  # size(1,1,13,13)

    stride_h = input_size[0] / grid_h_size
    stride_w = input_size[1] / grid_w_size

    scaled_anchors = float_tensor([(a_h / stride_h, a_w / stride_w) for a_h, a_w in
                                   anchors])  # self.anchors=[(144, 174), (195, 227), (264, 337)],scaled_anchors=[[ 4.5000,  5.4375],[ 6.0938,  7.0938],[ 8.2500, 10.5312]]
    # 注意: anchor_h/anchor_w为anchor在grid中的高宽，而非相对网络输入的宽高，这里做了变换
    anchor_h = scaled_anchors[:, 0:1].view((1, len(anchors), 1, 1))  # size(1,3,1,1),取设置的预测框初识高
    anchor_w = scaled_anchors[:, 1:2].view((1, len(anchors), 1, 1))  # size(1,3,1,1),取设置的预测框初识宽

    pred_boxes = float_tensor(prediction[..., :4].shape)  # pred_boxes.size=(1,3,13,13,4)，最后一维依次为(x,y,h,w)
    pred_boxes[..., 0] = x.data + grid_x  # bx=sigmoid(tx) + cx
    pred_boxes[..., 1] = y.data + grid_y  # by=sigmoid(ty) + cy
    pred_boxes[..., 2] = torch.exp(h.data) * anchor_h  # bh=ph * e^th (ph=anchor_h)
    pred_boxes[..., 3] = torch.exp(w.data) * anchor_w  # bw=pw * e^tw (pw=anchor_w)

    output = torch.cat(
        (
            pred_boxes.view(num_samples, -1, 4) * float_tensor([stride_w, stride_h, stride_h, stride_w]),
            # 由(1,3,13,13,4)变为(1,507,4)*32
            pred_conf.view(num_samples, -1, 1),  # 由(1,3,13,13)变为(1,507,1)
            pred_cls.view(num_samples, -1, num_classes),  # 由(1,3,13,13,44)变为(1,507,44)
        ),
        -1,
    )  # (1,507,49),最后一维=(x,y,h,w)+1置信度+44分类

    if labels is None:
        return output, 0, {}
    else:
        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
            pred_boxes=pred_boxes,
            pred_cls=pred_cls,
            targets=labels,
            anchors=scaled_anchors,
            ignore_thres=ignore_threshold,
        )

        obj_mask = obj_mask.bool()
        noobj_mask = noobj_mask.bool()

        # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = torch.nn.MSELoss()(x[obj_mask], tx[obj_mask])
        loss_y = torch.nn.MSELoss()(y[obj_mask], ty[obj_mask])
        loss_w = torch.nn.MSELoss()(w[obj_mask], tw[obj_mask])
        loss_h = torch.nn.MSELoss()(h[obj_mask], th[obj_mask])
        loss_conf_obj = torch.nn.BCELoss()(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = torch.nn.BCELoss()(pred_conf[noobj_mask], tconf[noobj_mask])
        # loss_conf = obj_scale * loss_conf_obj + noobj_scale * loss_conf_noobj
        loss_cls = torch.nn.BCELoss()(pred_cls[obj_mask], tcls[obj_mask])
        total_loss = coord_scale * (loss_x + loss_y + loss_w + loss_h) + \
                     obj_scale * loss_conf_obj + \
                     noobj_scale * loss_conf_noobj + \
                     cls_scale * loss_cls

        # Metrics
        cls_acc = 100 * class_mask[obj_mask].mean()
        conf_obj = pred_conf[obj_mask].mean()
        conf_noobj = pred_conf[noobj_mask].mean()
        conf50 = (pred_conf > 0.5).float()
        iou50 = (iou_scores > 0.5).float()
        iou75 = (iou_scores > 0.75).float()
        detected_mask = conf50 * class_mask * tconf
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        metrics = {
            "grid_size": (grid_h_size, grid_w_size),
            "total_loss": to_cpu(total_loss).item(),
            "loss_x": to_cpu(coord_scale * loss_x).item(),
            "loss_y": to_cpu(coord_scale * loss_y).item(),
            "loss_w": to_cpu(coord_scale * loss_w).item(),
            "loss_h": to_cpu(coord_scale * loss_h).item(),
            "loss_conf_obj": to_cpu(obj_scale * loss_conf_obj).item(),
            "loss_conf_noobj": to_cpu(noobj_scale * loss_conf_noobj).item(),
            "loss_cls": to_cpu(cls_scale * loss_cls).item(),
            "cls_acc": to_cpu(cls_acc).item(),
            "recall50": to_cpu(recall50).item(),
            "recall75": to_cpu(recall75).item(),
            "precision": to_cpu(precision).item(),
            "conf_obj": to_cpu(conf_obj).item(),
            "conf_noobj": to_cpu(conf_noobj).item(),
        }

        return output, total_loss, metrics


def build_targets(pred_boxes, pred_cls, targets, anchors, ignore_thres):
    """
    根据标签, 构建网络需要的target
    :param pred_boxes: shape=(n,num_anchor,grid_h_size,grid_w_size,4), 最后一维依次为(box_x_center,box_y_center,box_h,box_w)
    :param pred_cls: shape=(n,num_anchor,grid_h_size,grid_w_size,num_classes), 最后一维为类别的one-hot
    :param targets: shape=(box_num,6), 其中6=(n,class_index,box_x_center,box_y_center,box_h,box_w)
    :param anchors: 预选框高宽[(h1,w1), (h2, w2), ......], 其size=num_anchor, 示例[(41,34), (72,76), (101,195), (151,93), (182,309), (270,159), (296,372), (373,265), (412,406)]
    :param ignore_thres: 范围:(0-1), 当iou大于该值, noobj_mask标记单元格有对象
    :return:
    """
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    float_tensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)  # n
    nA = pred_boxes.size(1)  # num_anchor
    nC = pred_cls.size(-1)  # num_classes
    nGh = pred_boxes.size(2)  # grid_h_size
    nGw = pred_boxes.size(3)  # grid_w_size

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nGh, nGw).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nGh, nGw).fill_(1)
    class_mask = float_tensor(nB, nA, nGh, nGw).fill_(0)
    iou_scores = float_tensor(nB, nA, nGh, nGw).fill_(0)
    tx = float_tensor(nB, nA, nGh, nGw).fill_(0)
    ty = float_tensor(nB, nA, nGh, nGw).fill_(0)
    tw = float_tensor(nB, nA, nGh, nGw).fill_(0)
    th = float_tensor(nB, nA, nGh, nGw).fill_(0)
    tcls = float_tensor(nB, nA, nGh, nGw, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = targets[:, 2:6] * float_tensor(
        [nGw, nGh, nGh, nGw])  # shape=(box_num,(box_x_center,box_y_center,box_h,box_w))
    gxy = target_boxes[:, :2]  # shape=(box_num,(box_x_center,box_y_center))
    ghw = target_boxes[:, 2:]  # shape=(box_num,(box_h,box_w))
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, ghw) for anchor in anchors])  # shape=(num_anchor,box_num)
    best_ious, best_n = ious.max(0)  # shape=(box_num), best_ious是最大值的Tensor, best_n是最大值对应的index的Tensor
    # Separate target values
    # (box_num,(n,class_index))=>((n,class_index),box_num).
    # 返回值中b为batch_size值shape=(box_num), target_labels为分类索引shape=(box_num)
    b, target_labels = targets[:, :2].long().t()
    gx, gy = gxy.t()  # gxy.t().shape=((box_x_center,box_y_center),box_num)
    gh, gw = ghw.t()  # ghw.t().shape=((box_h,box_w),box_num)
    gi, gj = gxy.long().t()  # gxy.long().t().shape=((box_x_grid_index,box_y_grid_index),box_num)
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        # noobj_mask.shape=(n, num_anchor, grid_h_size, grid_w_size)
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    # anchors.shape=(num_anchor,(h,w)), best_n.shape=(box_num), anchors[best_n] = (box_num,(h,w)), best_n的值为num_anchor索引
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
