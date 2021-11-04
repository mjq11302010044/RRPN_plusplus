# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np
from .bounding_box import RBoxList

from maskrcnn_benchmark.layers import nms as _box_nms
from rotation.rotate_polygon_nms import rotate_gpu_nms
from rotation.rbbox_overlaps import rbbx_overlaps
#from rotation.lanms import merge_quadrangle_n9

def rbox2poly(rboxes):

    ctr_x = rboxes[:, 0:1]
    ctr_y = rboxes[:, 1:2]
    width = rboxes[:, 2:3]
    height = rboxes[:, 3:4]
    angle = rboxes[:, 4:]

    # struct = np.zeros_like(rboxes[:, 0])

    l = (- width / 2.0)
    r = (width / 2.0)
    t = (- height / 2.0)
    b = (height / 2.0)

    # anti-clockwise [n, 1, 1]
    cosA = np.cos(-angle / 180 * np.pi)[..., np.newaxis]
    sinA = np.sin(-angle / 180 * np.pi)[..., np.newaxis]

    polys = np.concatenate([l, t, r, t, r, b, l, b], axis=1).reshape(-1, 4, 2)

    # [n, 4, 1]
    x_poly, y_poly = polys[..., 0:1], polys[..., 1:2]

    x_poly_new = x_poly * cosA - y_poly * sinA + ctr_x[..., np.newaxis]
    y_poly_new = x_poly * sinA + y_poly * cosA + ctr_y[..., np.newaxis]

    return np.concatenate([x_poly_new, y_poly_new], axis=-1).reshape(-1, 8)


def poly2rbox(qboxes, eps=1e-8):

    # qboxes: [K, 8(lt, rt, rb, lb)]
    # return rboxes: [N, (x_c, y_c, h, w, theta)]

    edge1 = np.sqrt((qboxes[:, 0] - qboxes[:, 2]) * (qboxes[:, 0] - qboxes[:, 2]) + (qboxes[:, 1] - qboxes[:, 3]) * (qboxes[:, 1] - qboxes[:, 3]))
    edge2 = np.sqrt((qboxes[:, 2] - qboxes[:, 4]) * (qboxes[:, 2] - qboxes[:, 4]) + (qboxes[:, 3] - qboxes[:, 5]) * (qboxes[:, 3] - qboxes[:, 5]))

    edge_com = np.concatenate([edge1[:, np.newaxis], edge2[:, np.newaxis]], axis=-1)
    height = np.min(edge_com, axis=-1)
    width = np.max(edge_com, axis=-1)

    arg_width = np.argmax(edge_com, axis=-1)

    angle1 = -np.arctan((qboxes[:, 3] - qboxes[:, 1]) / (qboxes[:, 2] - qboxes[:, 0] + eps)) / 3.1415926 * 180
    angle2 = -np.arctan((qboxes[:, 5] - qboxes[:, 3]) / (qboxes[:, 4] - qboxes[:, 2] + eps)) / 3.1415926 * 180
    angle_com = np.concatenate([angle1[:, np.newaxis], angle2[:, np.newaxis]], axis=-1)

    angle = angle_com[range(angle2.shape[0]), arg_width]

    ctr_cood = np.mean(qboxes.reshape(-1, 4, 2), axis=1)

    return np.concatenate([ctr_cood, width[:, np.newaxis], height[:, np.newaxis], angle[:, np.newaxis]], axis=-1)


def eastbox2rbox(eastbox, base_size, feature_size, scale, eps=1e-8):

    # eastbox: [K, sigmoid(top, right, bottom, left, angle)]
    # return rboxes: [N, (x_c, y_c, h, w, theta)]

    H, W = feature_size
    device = eastbox.device

    top, right, bottom, left = eastbox[..., 0] * float(base_size), \
                               eastbox[..., 1] * float(base_size), \
                               eastbox[..., 2] * float(base_size), \
                               eastbox[..., 3] * float(base_size)
    angle_pred = eastbox[..., 4]

    # [0, 1] to [-45, 135]
    pred_a = (angle_pred - 0.5) * 180.
    pred_arc = (angle_pred - 0.5) * 3.141592652358979# / 2.
    pred_w = left + right
    pred_h = top + bottom

    p_grid = (np.mgrid[:H, :W][np.newaxis, ...].reshape(2, -1).T + 0.5) * float(1. / scale)
    p_grid_X, p_grid_Y = p_grid[:, 1], p_grid[:, 0]

    p_grid_X_th = torch.tensor(p_grid_X).float().to(device)
    p_grid_Y_th = torch.tensor(p_grid_Y).float().to(device)
    cos_A = torch.cos(-pred_arc)
    sin_A = torch.sin(-pred_arc)

    # [N, grid_size] - [N, grid_size]
    ctr_x_shift = (pred_w / 2. - left) * cos_A - (pred_h / 2. - top) * sin_A
    ctr_y_shift = (pred_w / 2. - left) * sin_A + (pred_h / 2. - top) * cos_A
    # print("ctr:", ctr_x_shift.shape, p_grid_X_th.shape)

    # [N, grid_size] + [1, grid_size]
    ctr_x = ctr_x_shift + p_grid_X_th[None, ...]
    ctr_y = ctr_y_shift + p_grid_Y_th[None, ...]

    # all_proposals: [N, grid_size, 5]

    # print("all_proposals shape:", ctr_x.shape, ctr_y.shape, pred_w.shape, pred_h.shape, pred_a.shape)

    pred_a_cl = pred_a + (pred_w < pred_h).float() * 90.

    pred_w_cl = pred_w * (pred_w >= pred_h).float() + pred_h * (pred_w < pred_h).float()
    pred_h_cl = pred_h * (pred_w >= pred_h).float() + pred_w * (pred_w < pred_h).float()

    all_proposals = torch.cat(
        [
            ctr_x[..., None],
            ctr_y[..., None],
            pred_w_cl[..., None],
            pred_h_cl[..., None],
            pred_a_cl[..., None]
        ],
        dim=-1
    )

    return all_proposals


def eastbox2rbox_np(eastbox, base_size, feature_size, scale, eps=1e-8):

    # eastbox: [K, sigmoid(top, right, bottom, left, angle)]
    # return rboxes: [N, (x_c, y_c, h, w, theta)]

    H, W = feature_size
    # device = eastbox.device

    top, right, bottom, left = eastbox[..., 0] * base_size, \
                               eastbox[..., 1] * base_size, \
                               eastbox[..., 2] * base_size, \
                               eastbox[..., 3] * base_size
    angle_pred = eastbox[..., 4]

    # [0, 1] to [-45, 135]
    pred_a = (angle_pred - 0.5) * 180.
    pred_arc = (angle_pred - 0.5) * 3.141592652358979# / 2.
    pred_w = left + right
    pred_h = top + bottom

    p_grid = (np.mgrid[:H, :W][np.newaxis, ...].reshape(2, -1).T + 0.5) * (1. / scale)
    p_grid_X, p_grid_Y = p_grid[:, 1], p_grid[:, 0]

    p_grid_X_th = p_grid_X # torch.tensor(p_grid_X).float().to(device)
    p_grid_Y_th = p_grid_Y # torch.tensor(p_grid_Y).float().to(device)
    cos_A = np.cos(-pred_arc)
    sin_A = np.sin(-pred_arc)

    # [N, grid_size] - [N, grid_size]
    ctr_x_shift = (pred_w / 2. - left) * cos_A - (pred_h / 2. - top) * sin_A
    ctr_y_shift = (pred_w / 2. - left) * sin_A + (pred_h / 2. - top) * cos_A

    # print("ctr:", ctr_x_shift.shape, p_grid_X_th.shape)

    # [N, grid_size] + [1, grid_size]
    ctr_x = ctr_x_shift + p_grid_X_th[np.newaxis, ...]
    ctr_y = ctr_y_shift + p_grid_Y_th[np.newaxis, ...]

    # all_proposals: [N, grid_size, 5]

    # print("all_proposals shape:", ctr_x.shape, ctr_y.shape, pred_w.shape, pred_h.shape, pred_a.shape)

    pred_a_cl = pred_a + (pred_w < pred_h) * 90.

    pred_w_cl = pred_w * (pred_w >= pred_h) + pred_h * (pred_w < pred_h)
    pred_h_cl = pred_h * (pred_w >= pred_h) + pred_w * (pred_w < pred_h)

    all_proposals = np.concatenate(
        [
            ctr_x[..., np.newaxis],
            ctr_y[..., np.newaxis],
            pred_w_cl[..., np.newaxis],
            pred_h_cl[..., np.newaxis],
            pred_a_cl[..., np.newaxis]
        ],
        axis=-1
    )

    return all_proposals


def set2rboxes(proposals):

    # pred_w = ch_boxes
    # for target in targets:
    ch_boxes = proposals.clone()

    gt_w = ch_boxes[:, 2]
    gt_h = ch_boxes[:, 3]
    gt_a = ch_boxes[:, 4]

    gt_a_cl = gt_a + (gt_w < gt_h).float() * 90.

    gt_w_cl = gt_w * (gt_w >= gt_h).float() + gt_h * (gt_w < gt_h).float()
    gt_h_cl = gt_h * (gt_w >= gt_h).float() + gt_w * (gt_w < gt_h).float()

    ch_boxes[:, 2] = gt_w_cl
    ch_boxes[:, 3] = gt_h_cl
    ch_boxes[:, 4] = gt_a_cl

    return ch_boxes

def cluster_nms(boxlist, nms_thresh, max_proposals=-1, score_field="score", GPU_ID=0):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maxium suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    if boxlist.has_field("mask"):
        print("Mask merge not supported yet... Return the input...")
        return boxlist

    # boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)

    ##################################################
    # convert to numpy before calculate
    boxes_np = boxes.data.cpu().numpy()
    score_np = score.data.cpu().numpy()

    polys_np = rbox2poly(boxes_np)
    polys_np = np.concatenate([polys_np, score_np.reshape(-1, 1)], axis=-1)
    polys_np = merge_quadrangle_n9(polys_np.astype('float32'), nms_thresh)

    if polys_np.shape[0] < 1:
        # Fake boxes
        rboxes_np = np.zeros((1, 5), np.float32)
        scores = np.zeros((1,), np.float32)
    else:
        rboxes_np = poly2rbox(polys_np[:, :8])
        scores = polys_np[:, -1]

    ret_boxlist = RBoxList(rboxes_np, boxlist.size)
    ret_boxlist.bbox = ret_boxlist.bbox.to(boxlist.bbox.device)
    ret_boxlist.add_field(score_field, torch.tensor(scores).to(boxlist.bbox.device))

    return ret_boxlist #.convert(mode)



def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="score", GPU_ID=0):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maxium suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist

    # boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)

    ##################################################
    # convert to numpy before calculate
    boxes_np = boxes.data.cpu().numpy()
    score_np = score.data.cpu().numpy()
    # keep = _box_nms(boxes, score, nms_thresh)
    ch_proposals = boxes_np.copy()
    ch_proposals[:, 2:4] = ch_proposals[:, 3:1:-1]
    # x,y,h,w,a

    # print('ch_proposals:',ch_proposals.shape)
    # print('score_np:', score_np.shape)

    if ch_proposals.shape[0] < 1:

        # Fake boxes
        rboxes_np = np.zeros((1, 5), np.float32)
        scores = np.zeros((1,), np.float32)

        ret_boxlist = RBoxList(rboxes_np, boxlist.size)
        ret_boxlist.bbox = ret_boxlist.bbox.to(boxlist.bbox.device)
        ret_boxlist.add_field(score_field, torch.tensor(scores).to(boxlist.bbox.device))

        return ret_boxlist

    keep = rotate_gpu_nms(np.array(np.hstack((ch_proposals, score_np[..., np.newaxis])), np.float32), nms_thresh, GPU_ID)  # D
    # print time.time() - tic
    if max_proposals > 0:
        keep = keep[:max_proposals]

    keep_th = torch.tensor(keep, dtype=torch.long).to(boxlist.bbox.device)

    # print('keep_th:', keep_th.type())
    ##################################################
    # proposals = proposals[keep, :]
    # scores = scores[keep]

    # if max_proposals > 0:
    #     keep = keep[:max_proposals]
    boxlist = boxlist[keep_th]

    # print('boxlist:', boxlist.bbox.type())

    return boxlist #.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywha_boxes = boxlist.bbox
    _, _, ws, hs, a_s = xywha_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2, GPU_ID=0):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,5].
      box2: (BoxList) bounding boxes, sized [M,5].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    eps = 1e-8

    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    if boxlist1.bbox.size()[0] < 1 or boxlist1.bbox.size()[0] < 1:
        raise RuntimeError(
                "boxlists should have size larger than 0, got {}, {}".format(boxlist1.bbox.size()[0], boxlist1.bbox.size()[0]))

    ###########################################################
    box1, box2 = boxlist1.bbox, boxlist2.bbox

    box1_np = box1.data.cpu().numpy()
    box2_np = box2.data.cpu().numpy()

    ch_box1 = box1_np.copy()
    ch_box1[:, 2:4] = ch_box1[:, 3:1:-1]
    ch_box2 = box2_np.copy()
    ch_box2[:, 2:4] = ch_box2[:, 3:1:-1]

    #ch_box2[:, 2:4] += 16

    overlaps = rbbx_overlaps(np.ascontiguousarray(ch_box1, dtype=np.float32),
                             np.ascontiguousarray(ch_box2, dtype=np.float32), GPU_ID)

    #print('ch_box shape:', ch_box1.shape, ch_box2.shape)
    #print('ch_box shape:', ch_box1[:, 2:4], ch_box2[:, 2:4], ch_box2[:, 4])
    #print('overlaps_shape:', overlaps.shape)
    #print('overlaps:', np.unique(overlaps)[:10], np.unique(overlaps)[-10:])
    ############################################
    # Some unknown bug on complex coordinate
    overlaps[overlaps > 1.00000001] = 0.0
    ############################################

    ###########################################################
    '''
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    '''
    # print('bbox_shape_monitor:', overlaps.shape, boxlist1.bbox.device, boxlist2.bbox.device)

    overlaps_th = torch.tensor(overlaps).to(boxlist1.bbox.device)

    return overlaps_th


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]

    tensors = [ten.type_as(tensors[0]) for ten in tensors]
    return torch.cat(tensors, dim)


def _cat_seq(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]

    max_len = -1
    for ten in tensors:
        if max_len < ten.shape[1]:
            max_len = ten.shape[1]

    new_tensors = []
    for ten in tensors:
        new_ten = torch.zeros((ten.shape[0], max_len)).type_as(ten)
        new_ten[:, :ten.shape[1]] = ten
        new_tensors.append(new_ten)

    return torch.cat(new_tensors, dim)



def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, RBoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    # print("In box:")
    cat_boxes = RBoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)
    # print("In box end")

    for field in fields:

        if field in ["words", "word"]:
            data = _cat_seq([bbox.get_field(field) for bbox in bboxes], dim=0)
        else:
            data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)

        cat_boxes.add_field(field, data)

    return cat_boxes
