import numpy as np
import cv2
import math


##############################
#                            #
#      ### RBox gt ###       #
#                            #
##############################

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


def pyramid_targets_rbox(imshape, scale_stack, area_thres, rboxes, dense_ratio=0.7, areas=None):
    p_heatmap_stack = []
    p_angle_stack = []
    p_target_stack = []

    # area_thres = [32 ** 2, 64 ** 2]

    scaled_qboxes = []
    scaled_rboxes = []

    box_in_4pts = rbox2poly(rboxes)

    if not areas is None:
        # small = areas < area_thres[0]
        medium = (areas >= area_thres[0]) & (areas < area_thres[1])
        # large = areas >= area_thres[1]
        # print('pyshape:', small.shape, box_in_4pts.shape)
        # scaled_qboxes.append(box_in_4pts)
        scaled_qboxes.append(box_in_4pts[medium])
        # scaled_qboxes.append(box_in_4pts)

        # scaled_rboxes.append(rboxes)  # [small]
        scaled_rboxes.append(rboxes[medium])  # [medium]
        # scaled_rboxes.append(rboxes)  # [large]

    for i in range(len(scale_stack)):
        scale = scale_stack[i]

        # if not areas is None:
        box_in_4pts = scaled_qboxes[i]
        rbox = scaled_rboxes[i]
        # print("scaled_qboxes:", scaled_qboxes[i], rbox)

        heatmap, target = make_target_rbox(imshape, scale, box_in_4pts, rbox, dense_ratio=dense_ratio)
        p_heatmap_stack.append(heatmap)
        p_angle_stack.append([])
        p_target_stack.append(target)

    p_heatmap = np.concatenate(p_heatmap_stack, axis=0)
    p_target = np.concatenate(p_target_stack, axis=0)

    return p_heatmap, p_target


def make_target_rbox(imshape, scale, box_in_4pts, rboxes, dense_ratio=0.7):
    # heatmap_stack: [nbox, H, W]
    # target_stack: [nbox, 8, H, W]

    # print("box_in_4pts make:", box_in_4pts, rboxes)

    heatmap_stack, angle_cls, target_stack, angle_reg = compute_target_rbox(box_in_4pts, scale,
                                                                            get_heatmap_rbox(
                                                                                 box_in_4pts,
                                                                                 imshape,
                                                                                 float(1 / scale)
                                                                            ),
                                                                            rboxes[:, -1])

    # print("angle_cls:", angle_cls.shape)
    # print("angle_reg:", angle_reg.shape)

    # heatmap: [H*W]
    # target: [H*W, 4+6]

    target_stack = np.transpose(target_stack, (0, 2, 3, 1))

    # angle_cls = np.sum(angle_cls, axis=1)[:, np.newaxis, :, :]
    # print("angle_cls:", angle_cls.shape)
    # angle_cls = np.transpose(angle_cls, (0, 2, 3, 1))
    angle_reg = np.transpose(angle_reg, (0, 2, 3, 1))

    # print("heatmap_stack:", heatmap_stack.shape)
    heatmap = np.sum(heatmap_stack, axis=0).reshape(-1)
    # print("heatmap:", heatmap.shape, np.unique(heatmap))
    heatmap[heatmap > 1] = 0

    # 4 edges
    target = np.sum(target_stack, axis=0).reshape(-1, 4)
    # print("make_target:", np.where(target[:, 0] != 0), target[target[:, 0] != 0])
    angle_cls_map = angle_cls  # np.sum(angle_cls, axis=0).reshape(-1)
    # remove overlapping labels
    # angle_cls_map[angle_cls_map > 6] = 0

    # print("angle_reg_map:", angle_reg.shape)
    angle_reg_map = np.sum(angle_reg, axis=0).reshape(-1, 1)

    # print("[heatmap, angle_cls_map]", heatmap.shape, angle_cls_map.shape)

    # 1 channel for textness, 6 for angle cls, [b, 1, h, w]
    # heatmap = np.concatenate([heatmap, angle_cls_map], axis=0)
    # 4 channels for coods reg, 6 for angle reg, [b, 4, h, w]
    # print("[target, angle_reg_map]", target.shape, angle_reg_map.shape)
    target = np.concatenate([target, angle_reg_map], axis=1)

    # print("make_target:", heatmap.shape, target.shape)
    target = target * heatmap[..., np.newaxis]

    return heatmap, target


def make_target_rbox_mc(imshape, scale, box_in_4pts, rboxes, seq_labels, dense_ratio=0.7):
    # heatmap_stack: [nbox, H, W]
    # target_stack: [nbox, 8, H, W]

    # print("box_in_4pts make:", box_in_4pts, rboxes)
    # gt_boxes, imshape, stride, gt_rboxes, seq_labels
    heatmap, classmap = get_heatmap_rbox_multiclass(
        box_in_4pts,
        imshape,
        float(1 / scale),
        rboxes,
        seq_labels
    )

    heatmap_stack, angle_cls, target_stack, angle_reg = compute_target_rbox(box_in_4pts,
                                                                            scale,
                                                                            heatmap,
                                                                            rboxes[:, -1])



    # print("angle_cls:", angle_cls.shape)
    # print("angle_reg:", angle_reg.shape)

    # heatmap: [H*W]
    # classmap: [H*W]
    # target: [H*W, 4+6]

    target_stack = np.transpose(target_stack, (0, 2, 3, 1))

    # angle_cls = np.sum(angle_cls, axis=1)[:, np.newaxis, :, :]
    # print("angle_cls:", angle_cls.shape)
    # angle_cls = np.transpose(angle_cls, (0, 2, 3, 1))
    angle_reg = np.transpose(angle_reg, (0, 2, 3, 1))

    # print("heatmap_stack:", heatmap_stack.shape)
    heatmap = np.sum(heatmap_stack, axis=0).reshape(-1)

    biclsmap = np.sum((classmap > 0).astype(np.float32), axis=0).reshape(-1)
    classmap = np.sum(classmap, axis=0).reshape(-1)

    # print("heatmap:", heatmap.shape, np.unique(heatmap))
    heatmap[heatmap > 1] = 0
    classmap[biclsmap > 1] = 0

    # 4 edges
    target = np.sum(target_stack, axis=0).reshape(-1, 4)
    # print("make_target:", np.where(target[:, 0] != 0), target[target[:, 0] != 0])
    angle_cls_map = angle_cls  # np.sum(angle_cls, axis=0).reshape(-1)
    # remove overlapping labels
    # angle_cls_map[angle_cls_map > 6] = 0

    # print("angle_reg_map:", angle_reg.shape)
    angle_reg_map = np.sum(angle_reg, axis=0).reshape(-1, 1)

    # print("[heatmap, angle_cls_map]", heatmap.shape, angle_cls_map.shape)

    # 1 channel for textness, 6 for angle cls, [b, 1, h, w]
    # heatmap = np.concatenate([heatmap, angle_cls_map], axis=0)
    # 4 channels for coods reg, 6 for angle reg, [b, 4, h, w]
    # print("[target, angle_reg_map]", target.shape, angle_reg_map.shape)
    target = np.concatenate([target, angle_reg_map], axis=1)

    # print("make_target:", heatmap.shape, target.shape)
    target = target * heatmap[..., np.newaxis]

    return heatmap, target, classmap



def get_heatmap_rbox(gt_boxes, imshape, stride, proportion=0.7):
    # gt_boxes_4pts:[n, (lt, rt, rb, lb) * (x, y)] gt box pts in anti-clock order within 8 channels

    cls_num = 1
    fill_mask_ori = np.zeros((math.ceil(imshape[0]), math.ceil(imshape[1])))
    # print("fill_mask_ori:", fill_mask_ori.shape)
    mask_stack = []
    if len(gt_boxes) < 1:
        mask_stack.append(fill_mask_ori[np.newaxis, ...])

    for i in range(len(gt_boxes)):
        fill_mask = fill_mask_ori.copy()
        coods = np.array(gt_boxes[i], np.int32).reshape(4, 2)

        pt1 = coods[0]
        pt2 = coods[1]
        pt3 = coods[2]
        pt4 = coods[3]

        ctr = (((pt1 + pt3) / 2 + (pt2 + pt4) / 2) / 2).reshape(-1, 2)
        rescale_coods = np.array((coods - ctr) * proportion + ctr, np.int32)

        fill_mask = cv2.fillPoly(fill_mask, np.array(np.array([rescale_coods]) / stride, np.int32), cls_num)

        # print("rescale_coods:", np.array(np.array([rescale_coods]) / stride, np.int32), fill_mask.shape)

        mask_stack.append(fill_mask[np.newaxis, ...])

    return np.concatenate(mask_stack, axis=0)


def get_heatmap_rbox_multiclass(gt_boxes, imshape, stride, gt_rboxes, seq_labels, proportion=0.7):
    # gt_boxes_4pts:[n, (lt, rt, rb, lb) * (x, y)] gt box pts in anti-clock order within 8 channels

    cls_num = 1
    fill_mask_ori = np.zeros((math.ceil(imshape[0]), math.ceil(imshape[1])))
    char_mask_ori = np.zeros((math.ceil(imshape[0]), math.ceil(imshape[1])))

    # print("fill_mask_ori:", fill_mask_ori.shape)
    mask_stack = []
    char_map_stack = []
    if len(gt_boxes) < 1:
        mask_stack.append(fill_mask_ori[np.newaxis, ...])
        char_map_stack.append(char_mask_ori[np.newaxis, ...])

    label_len = np.array([len(seq) for seq in seq_labels])

    char_poses = []
    for i in range(len(label_len)):
        l = label_len[i]
        rbox = gt_rboxes[i]

        x, y, w, h, a = rbox
        arc = -a * np.pi / 180.0

        w *= proportion
        h *= proportion

        # radius in [1, w / 2]
        char_r = min(max(rbox[2] / float(2 * l + 1e-10), 1), w / 3.) / stride

        xs = [x - (w / 2 - w / (2 * l) - nth * (w / l)) * np.abs(np.cos(arc)) for nth in range(l)]
        ys = [y - (w / 2 - w / (2 * l) - nth * (w / l)) * np.abs(np.sin(arc)) for nth in range(l)]

        # [x, y, r, label]
        char_poses.append([[xs[n] / stride, ys[n] / stride, char_r, seq_labels[i][n]] for n in range(l)])

    for i in range(len(gt_boxes)):
        fill_mask = fill_mask_ori.copy()
        char_mask = char_mask_ori.copy()

        coods = np.array(gt_boxes[i], np.int32).reshape(4, 2)

        pt1 = coods[0]
        pt2 = coods[1]
        pt3 = coods[2]
        pt4 = coods[3]

        ctr = (((pt1 + pt3) / 2 + (pt2 + pt4) / 2) / 2).reshape(-1, 2)
        rescale_coods = np.array((coods - ctr) * proportion + ctr, np.int32)

        fill_mask = cv2.fillPoly(fill_mask, np.array(np.array([rescale_coods]) / stride, np.int32), cls_num)

        char_pos = char_poses[i]
        for n in range(label_len[i]):

            char_mask = cv2.circle(
                char_mask,
                (int(char_pos[n][0]), int(char_pos[n][1])),
                int(char_pos[n][2]),
                int(char_pos[n][3]),
                -1
            )
        # print("char_mask:", np.unique(char_mask))

        mask_stack.append(fill_mask[np.newaxis, ...])
        char_map_stack.append(char_mask[np.newaxis, ...])

    return np.concatenate(mask_stack, axis=0), np.concatenate(char_map_stack, axis=0)


def compute_target_rbox(gt_boxes_4pts, scale, heatmap, angles, base_angle=30.):
    # gt_boxes_4pts: qbox in clock-wise
    h, w = heatmap.shape[1:]

    if gt_boxes_4pts.shape[0] < 1:
        return heatmap[np.newaxis, ...], [], np.zeros((1, 4, h, w)), np.zeros((1, 1, h, w))

    # p_grid in [x, y] shape
    p_grid = (np.mgrid[:h, :w][np.newaxis, ...].reshape(2, -1).T + 0.5) * float(1. / scale)
    p_grid = np.concatenate([p_grid[:, 1:2], p_grid[:, 0:1]], axis=-1)

    gt_boxes_4pts = np.array(gt_boxes_4pts).reshape(-1, 4, 2).astype(np.float32)

    pj_dis_coll = []

    for i in range(gt_boxes_4pts.shape[0]):

        A = gt_boxes_4pts[i]
        B = np.concatenate([gt_boxes_4pts[i][1:], gt_boxes_4pts[i][0:1]], axis=0)

        # AB: [4, 2]
        AB = B - A

        # AP: [line, grid, cood] -> [4, h * w, 2]
        AP = p_grid[np.newaxis, :, :] - gt_boxes_4pts[i][:, np.newaxis, :]
        AB_norm = np.sqrt(np.sum(AB ** 2, axis=-1))[..., np.newaxis]
        # AP_norm = np.sqrt(np.sum(AP ** 2, axis=-1))[..., np.newaxis]

        '''
        # print("AP_norm * sin_BAP:", AB.shape, AP.shape, AB_norm.shape, AP_norm.shape, np.tile(AB, (AP.shape[0], 1)).shape)
        cos_BAP = np.abs(np.sum(AB[:, np.newaxis, :] * AP, axis=-1))
        BAP_fraction = (AB_norm[:, np.newaxis, :] * AP_norm)
        # print("BAP_fraction:", cos_BAP.shape, BAP_fraction.shape)
        cos_BAP = cos_BAP[:, :, np.newaxis] / (BAP_fraction + 1e-10)
        sin_BAP = np.sqrt(1 - cos_BAP ** 2)
        # print("AP_norm * sin_BAP:", AP_norm.shape, sin_BAP.shape)

        # norm for each level by scale
        pj_dis = AP_norm * sin_BAP # * scale * (0.5 ** 3)
        '''

        # [4, 1]
        X1, Y1 = AB[..., 0:1], AB[..., 1:2]
        # [4, h * w, 1]
        X2, Y2 = AP[..., 0:1], AP[..., 1:2]

        dis_numerator = np.abs(X1[:, np.newaxis, :] * Y2 - X2 * Y1[:, np.newaxis, :])
        pj_dis = dis_numerator / (AB_norm[:, np.newaxis, :] + 1e-10)

        pj_dis_coll.append(pj_dis.reshape(AB.shape[0], h, w)[np.newaxis, ...])

    for i in range(heatmap.shape[0]):
        pj_dis_coll[i] *= heatmap[i]

    # Angle Map
    angles = np.array(angles)

    # angle_cls values ranged in [-1, 4], 6 in total
    # angles_cls = np.round(angles / base_angle)
    angles_reg = angles / 180. * np.pi  # - angles_cls * base_angle
    # angles_cls += 2

    angle_clss = []
    angle_regs = []

    for i in range(heatmap.shape[0]):
        # 0 to be background
        # angle_cls_map = np.zeros((int(180 / base_angle) + 1, h, w))
        angle_reg_map = np.zeros((1, h, w))

        # A_cls_map = heatmap[i] * angles_cls[i]
        # print("A_cls_map:", heatmap[i].shape, np.unique(angles_reg[i]), angles_reg[i].shape)
        A_reg_map = heatmap[i] * angles_reg[i]

        # print("A_cls_map:", A_cls_map.shape, np.unique(A_cls_map), np.unique(A_reg_map))

        # angle_cls_map[int(angles_cls[i])] = A_cls_map
        angle_reg_map[0] = A_reg_map

        # angle_clss.append(angle_cls_map[np.newaxis, ...])
        angle_regs.append(angle_reg_map[np.newaxis, ...])

    # print("angle_regs:", np.concatenate(angle_regs, axis=0).shape)
    # pj has a shape of [4, h, w]
    return heatmap, angle_clss, np.concatenate(pj_dis_coll, axis=0), np.concatenate(angle_regs, axis=0)


def rbox_transform_inv(target, imsize):
    # print("target:", target.shape)

    H, W = imsize

    # mul_scale = scale * (0.5 ** 3)

    target[..., 0] = target[..., 0] * float(H)  # / 1.4
    target[..., 1] = target[..., 1] * float(W)  # / 1.4
    target[..., 2] = target[..., 2] * float(H)  # / 1.4
    target[..., 3] = target[..., 3] * float(W)  # / 1.4

    return target


def get_rproposals(tarmap, heatmap, angle_map, scale, thres=0.4, imsize=()):
    # tarmap: [1, (lt, rt, rb, lb), H, W]
    # print(tarmap.shape)

    # print("rproposal:", tarmap.shape, heatmap.shape)

    ph, pw = heatmap.shape

    tarmap = tarmap.reshape(ph * pw, -1)

    # heatmap: [H, W]
    boundry_map = tarmap[:, :4]
    angle_reg = tarmap[:, 4:].reshape(-1, 1)

    proposal_map = rbox_transform_inv(boundry_map, imsize)

    scores = heatmap.reshape(-1)

    angle_final = (angle_reg - 0.5) * 90.
    final_proposals = np.hstack([proposal_map, angle_final, scores[..., np.newaxis]])
    # return: [ph, pw, 4 + 1 + 1]

    return final_proposals.reshape(ph, pw, -1)


def decoding_rboxes(proposals, scales):
    # proposals: [d1, d2, d3, d4, angles, scores]

    ph, pw = proposals.shape[:2]
    proposals = proposals.reshape(ph * pw, -1)

    # print("decoding_rboxes:", ph, pw, scales)

    # cood_grid = np.mgrid[:ph, :pw] / scales

    # p_grid in [x, y] shape
    p_grid = (np.mgrid[:ph, :pw][np.newaxis, ...].reshape(2, -1).T + 0.5) * float(1. / scales)
    p_grid = np.concatenate([p_grid[:, 1:2], p_grid[:, 0:1]], axis=-1)

    d1, d2, d3, d4 = proposals[..., 0:1], proposals[..., 1:2], proposals[..., 2:3], proposals[..., 3:4]
    angles_pred = proposals[..., 4:5]

    cos_A = np.cos(-angles_pred * np.pi / 180.0)
    sin_A = np.sin(-angles_pred * np.pi / 180.0)

    # print("sin_A:", sin_A.shape, d1.shape)

    od1 = np.concatenate([d1 * sin_A, -d1 * cos_A], axis=-1)
    od2 = np.concatenate([d2 * cos_A, d2 * sin_A], axis=-1)
    od3 = np.concatenate([-d3 * sin_A, d3 * cos_A], axis=-1)
    od4 = np.concatenate([-d4 * cos_A, -d4 * sin_A], axis=-1)

    cood_grid = p_grid
    # cood_grid[:, 0], cood_grid[:, 1] = cood_grid[:, 1], cood_grid[:, 0]

    pt1 = od1 + od2 + cood_grid
    pt2 = od2 + od3 + cood_grid
    pt3 = od3 + od4 + cood_grid
    pt4 = od4 + od1 + cood_grid

    # [lt, rt, rb, lb, scores]
    boxes_in_quad = np.concatenate([pt4, pt1, pt2, pt3, proposals[:, 5:]], axis=-1)

    return boxes_in_quad


def draw_rboxes(proposals, image, scales, confident=0.7, debug=False, gt=True):
    # get cood only
    # gt_boxes_coods = gt_boxes[:, :8]
    # gt_boxes_coods = gt_boxes_coods.reshape(-1, 4, 2).astype(np.int64)
    # print('draw_boxes:', gt_boxes.shape, gt_boxes[:10])
    # image = cv2.polylines(image.copy(), gt_boxes_coods, True, 255)

    # proposals: [ph, pw, dis * 4 + angle * 1 + scores]

    ph, pw = proposals.shape[:2]

    proposals = proposals.reshape(ph * pw, -1)

    # cood_grid = np.mgrid[:ph, :pw] / scales

    # p_grid in [x, y] shape
    p_grid = (np.mgrid[:ph, :pw][np.newaxis, ...].reshape(2, -1).T + 0.5) * float(1. / scales)
    p_grid = np.concatenate([p_grid[:, 1:2], p_grid[:, 0:1]], axis=-1)

    d1, d2, d3, d4 = proposals[..., 0:1], proposals[..., 1:2], proposals[..., 2:3], proposals[..., 3:4]
    angles_pred = proposals[..., 4:5]

    cos_A = np.cos(-angles_pred * np.pi / 180.0)
    sin_A = np.sin(-angles_pred * np.pi / 180.0)

    # print("sin_A:", sin_A.shape, d1.shape)

    od1 = np.concatenate([d1 * sin_A, -d1 * cos_A], axis=-1)
    od2 = np.concatenate([d2 * cos_A, d2 * sin_A], axis=-1)
    od3 = np.concatenate([-d3 * sin_A, d3 * cos_A], axis=-1)
    od4 = np.concatenate([-d4 * cos_A, -d4 * sin_A], axis=-1)

    cood_grid = p_grid
    # cood_grid[:, 0], cood_grid[:, 1] = cood_grid[:, 1], cood_grid[:, 0]

    pt1 = od1 + od2 + cood_grid
    pt2 = od2 + od3 + cood_grid
    pt3 = od3 + od4 + cood_grid
    pt4 = od4 + od1 + cood_grid

    if debug:
        pt1_vis = od1 + od2
        print("pt1_vis:", pt1_vis.shape, cood_grid.shape)
        print("pt1_vis:", pt1_vis[pt1_vis[:, 0] != 0], cood_grid[pt1_vis[:, 0] != 0])
        print("cood_grid:", cood_grid.shape, od4.shape, (d1 * cos_A).shape)
        print("p_grid in draw:", p_grid.shape, p_grid[:10], proposals[proposals[:, 0] != 0].astype(np.int32))

    boxes_in_quad = np.concatenate([pt1, pt2, pt3, pt4, proposals[:, 5:]], axis=-1)

    choosen_boxes = boxes_in_quad[boxes_in_quad[:, -1] > confident]

    ch_proposals = proposals[boxes_in_quad[:, -1] > confident]
    argsorted = np.argsort(-ch_proposals[:, -1])

    ch_proposals = ch_proposals[argsorted]
    choosen_boxes = choosen_boxes[argsorted]

    if gt:
        colors = [
            (255, 0, 0),
            (255, 153, 0),
            (255, 255, 0),
            (0, 255, 0)
        ]
    else:
        colors = [
            (0, 0, 255),
            (0, 0, 255),
            (0, 0, 255),
            (0, 0, 255)
        ]

    for i in range(choosen_boxes.shape[0]):
        box = choosen_boxes[i][:8].reshape(4, 2)
        box = box.astype(np.int32)
        for j in range(box.shape[0]):
            cv2.line(image, (box[j][0], box[j][1]), (box[(j + 1) % box.shape[0]][0], box[(j + 1) % box.shape[0]][1]),
                     colors[j], 1)

    return image


if __name__ == '__main__':
    gt_boxes = np.array([
        [10, 10, 50, 10, 50, 90, 10, 90],
        [110, 110, 150, 110, 170, 190, 130, 190]
    ])
    feat_map = np.zeros((1, 3, 500, 500))

    scale = 1

    heatmap = get_heatmap(gt_boxes, feat_map.shape[2:], 1 / scale, proportion=0.7)

    print('pre_heatmap:', heatmap.shape, np.unique(heatmap[0].astype(np.uint8)))
    cv2.imshow('win', np.sum(heatmap, 0).astype(np.uint8) * 255)
    cv2.waitKey(0)
    heatmap, angle_cls, target, angle_reg = compute_target_rbox(gt_boxes * scale, heatmap, [95, 28])

    cv2.imshow("win", draw_boxes(gt_boxes, np.transpose(feat_map[0], (1, 2, 0))))
    print('main_heatmap:', heatmap.shape)
    cv2.imshow("heatmap:", heatmap.reshape(heatmap.shape[1] * 2, heatmap.shape[2]))
    print('main_target:', target.shape)

    # cood: 0, 30, 70
    cood_target = target[0, :, 22, 19]
    print("cood_target 4 0,22,19:", cood_target.shape, cood_target)

    show_target = np.transpose(target, (0, 2, 1, 3)).reshape(target.shape[2] * 2, target.shape[3] * 4)
    print('show_target:', np.unique(show_target))
    cv2.imshow("target:", ((show_target != 0) * 255).astype(np.uint8))
    cv2.waitKey(0)