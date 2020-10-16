# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.rboxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.rboxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.rboxlist_ops import eastbox2rbox, eastbox2rbox_np

from maskrcnn_benchmark.utils.visualize import vis_image

import numpy as np
from .geo_target import make_target_rbox, rbox2poly, make_target_rbox_mc
import cv2
from PIL import Image

DEBUG=False

class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self,
                 proposal_matcher,
                 fg_bg_sampler,
                 box_coder,
                 base_size=640.,
                 size_stack=(32, 64, 128, 256),
                 scale_stack=(0.25, 0.125, 0.0625, 0.03125, 0.015625)):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.base_size = base_size
        self.size_stack = size_stack
        self.scale_stack = scale_stack

    def match_targets_to_anchors(self, anchor, target):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields([])
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, level_objectness, targets, size_range=None, scale=None, level=-1, boxes_reg=None):
        labels = []
        regression_targets = []
        classmaps = []

        N, C, H, W = level_objectness.shape
        device = level_objectness.device

        assert len(size_range) == 2, "size_range should be with left bound and right bound"

        for targets_per_image, boxes_reg_per_image in zip(targets, boxes_reg):
            # print("targets_per_image:", targets_per_image)
            # targets_per_image = targets_per_image_[0]

            with torch.no_grad():
                rboxes = targets_per_image.bbox.data.cpu().numpy()

                words_seq_np_ = targets_per_image.get_field("words").data.cpu().numpy()
                words_len_np_ = targets_per_image.get_field("word_length").data.cpu().numpy()

                box_areas = rboxes[:, 2] * rboxes[:, 3]
                bound_all = box_areas >= size_range[0]

                if size_range[1] != 0:
                    right_bound = box_areas < size_range[1]
                    bound_all = np.logical_and(bound_all, right_bound)
                rboxes_cp = rboxes[bound_all]

                words_seq_np = words_seq_np_[bound_all]
                words_len_np = words_len_np_[bound_all]

                word_seqs = []
                for i in range(len(words_len_np)):
                    word_seqs.append(words_seq_np[i][:words_len_np[i]])

                # print("rboxes:", rboxes_cp, size_range, box_areas)
                box_in_4pts = rbox2poly(rboxes_cp)
                # scale = 1 / np.rint(scale_w)

                # print("rboxes_cp:", rboxes_cp)

                # print("scale", scale)

                # heatmap: (H * W, 1)
                # target: (H * W, 4)

                heatmap, regression_targets_per_image, classmap = make_target_rbox_mc(
                    (H, W),
                    scale,
                    box_in_4pts,
                    rboxes_cp,
                    word_seqs,
                    dense_ratio=0.7
                )

                if DEBUG:

                    imW, imH = targets_per_image.size

                    if len(rboxes_cp) > 0:
                        # heatmap_np = heatmap.data.cpu().numpy()
                        # print("heatmap_np:", len(rboxes_cp), len(box_in_4pts), np.unique(heatmap))
                        heatmap_np = heatmap.reshape(H, W)
                        clsmap_np = classmap.reshape(H, W)
                        level_objectness_np = level_objectness.data.cpu().numpy()
                        level_objectness_np = level_objectness_np.reshape(H, W)

                        boxes_reg_per_image_np_ = boxes_reg_per_image.data.cpu().numpy()

                        class_map_logits = F.softmax(boxes_reg_per_image[5:].permute(1, 2, 0), dim=-1).data.cpu().numpy()
                        clsmap_prob = np.max(class_map_logits[..., 1:], axis=-1)
                        boxes_reg_per_image_np = boxes_reg_per_image_np_[:5]

                        print("boxes_reg_per_image_np:",
                              class_map_logits.shape,
                              boxes_reg_per_image_np.shape,
                              level_objectness_np.shape,
                              regression_targets_per_image.shape,
                              clsmap_prob.shape,
                              # np.unique(classmap),
                              # np.unique(heatmap_np),
                              # np.unique(clsmap_prob)[-10:]
                              )

                        # boxes_reg_per_image_np = boxes_reg_per_image_np.reshape(-1, 5)

                        cv2.imwrite("clsmap_prob_level_" + str(level) + ".jpg",
                                    ((clsmap_prob > 0.3) * 255).astype(np.uint8))
                        cv2.imwrite("heatmap_level_" + str(level) + ".jpg", ((heatmap_np > 0.7) * 255).astype(np.uint8))
                        cv2.imwrite("clsmap_level_" + str(level) + ".jpg", ((clsmap_np) * 3).astype(np.uint8))
                        cv2.imwrite("objectness_level_" + str(level) + ".jpg", ((level_objectness_np > 0.7) * 255).astype(np.uint8))

                        regression_targets_per_image_np = regression_targets_per_image.reshape(1, -1, 5).copy()

                        label = heatmap_np.reshape(-1) == 1
                        # regression_targets_per_image_np = regression_targets_per_image_np.reshape(1, -1, 5)

                        regression_targets_per_image_np[..., -1] = regression_targets_per_image_np[..., -1] / 3.14159265358979 * 2 + 0.5

                        gt_rboxes = eastbox2rbox_np(
                            regression_targets_per_image_np,
                            1,
                            (H, W),
                            scale
                            )

                        boxes_reg_per_image_np = np.transpose(boxes_reg_per_image_np, (1, 2, 0)).reshape(1, -1, 5)

                        picked_sig = boxes_reg_per_image_np.reshape(-1, 5)[label][..., -1]
                        # print("picked_sig:", np.unique(picked_sig[-5:]))

                        proposals = eastbox2rbox_np(
                            boxes_reg_per_image_np,
                            self.base_size,
                            (H, W),
                            scale
                        )

                        gt_rboxes = gt_rboxes.reshape(-1, 5)[label]
                        proposals = proposals.reshape(-1, 5)[label]
                        # boxes_reg_per_image_np = boxes_reg_per_image_np.reshape(-1, 5)[label]
                        # regression_targets_per_image_np = regression_targets_per_image_np.reshape(-1, 5)[label]

                        # print("recover:", gt_rboxes)
                        # print("proposals angle:", np.unique(proposals[:, -1][-5:]), np.unique(gt_rboxes[:, -1][-5:]))
                        # print("x, y gt pred:", gt_rboxes[:10][:, :4], proposals[:10][:, :4])
                        # print("t, r, n, l gt pred:", regression_targets_per_image_np[:10][:, :4], boxes_reg_per_image_np[:10][:, :4] * 640.)

                        canvas = np.zeros((imH, imW, 3)).astype(np.uint8)
                        canvas = Image.fromarray(canvas)

                        canvas = vis_image(canvas, gt_rboxes, mode=1)
                        canvas = vis_image(canvas, proposals, mode=2)

                        canvas.save("level_" + str(level) + "_boxes.jpg")

                        pass
                heatmap = torch.tensor(heatmap).to(device).float()
                classmap = torch.tensor(classmap).to(device).float()
                regression_targets_per_image = torch.tensor(regression_targets_per_image).to(device).float()

                # print("heatmap:", heatmap.shape, regression_targets_per_image.shape)

            labels.append(heatmap[None, ...])
            regression_targets.append(regression_targets_per_image[None, ...])
            classmaps.append(classmap[None, ...])

        return torch.cat(labels, dim=0), torch.cat(regression_targets, dim=0), torch.cat(classmaps, dim=0)

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        # anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        # labels, regression_targets = self.prepare_targets(anchors, targets)
        labels = []
        regression_targets = []
        classmaps = []

        objectness_flattened = []
        box_regression_flattened = []
        classmap_logit_flattened = []

        cnt = 0
        for objectness_per_level, box_regression_per_level in zip(
            objectness, box_regression
        ):
            N, A, H, W = objectness_per_level.shape
            # label: (N, grid_size, 1)
            # regression_target: (N, grid_size, 5)
            if cnt == 0:
                size_range = [0, self.size_stack[cnt] ** 2]
            elif cnt < len(self.size_stack):
                size_range = [self.size_stack[cnt - 1] ** 2, self.size_stack[cnt] ** 2]
            else:
                size_range = [self.size_stack[cnt - 1] ** 2, 0]

            label, regression_target, classmap = self.prepare_targets(
                objectness_per_level,
                targets,
                size_range,
                self.scale_stack[cnt],
                cnt+2,
                box_regression_per_level
            )

            # print("shape:", label.shape, regression_target.shape)

            labels.append(label[..., None])
            classmaps.append(classmap[..., None])
            regression_targets.append(regression_target)

            # print("objectness_per_level:", objectness_per_level.shape, box_regression_per_level.shape)

            # print("label", label.shape)
            objectness_per_level = objectness_per_level.permute(0, 2, 3, 1).view(
                N, H * W, 1
            )

            # box_regression_per_level = box_regression_per_level.view(N, -1, 4, H, W)
            # box_logits: [N, 5 + mc, H, W]

            box_regression_per_level = box_regression_per_level.permute(0, 2, 3, 1)
            # box_logits: [N, 5, H, W]
            box_regression_per_level_ = box_regression_per_level[..., :5].reshape(N, H * W, 5)
            # class_logits: [N, mc, H, W]
            classlogit_per_level = box_regression_per_level[..., 5:].reshape(N, H * W, -1)

            objectness_flattened.append(objectness_per_level)
            box_regression_flattened.append(box_regression_per_level_)
            classmap_logit_flattened.append(classlogit_per_level)

            # print("objectness_per_level:", objectness_per_level.shape, box_regression_per_level.shape)

            cnt += 1

        # concatenate on the first dimension (representing the feature levels), to
        # take into account the way the labels were generated (with all feature maps
        # being concatenated as well)
        objectness = cat(objectness_flattened, dim=1)
        box_regression = cat(box_regression_flattened, dim=1)
        classmap_logit = cat(classmap_logit_flattened, dim=1)

        N, grid, cls = classmap_logit.shape

        classmap_logit = classmap_logit.view(N * grid, cls)

        labels = torch.cat(labels, dim=1)
        regression_targets = torch.cat(regression_targets, dim=1)
        classmap = torch.cat(classmaps, dim=1).reshape(-1)

        # print("gt_shape:", labels.shape, regression_targets.shape, objectness.shape, box_regression.shape)

        if DEBUG:
            # labels_np = labels.data.cpu().numpy()
            # objn_np = objectness.data.cpu().numpy()
            # print("labels:", labels_np.shape, np.unique(labels_np)[-10:])
            # print("objectness:", objn_np.shape, np.unique(objn_np)[-10:])
            pass
        objectness_loss, box_loss = self.geo_loss(
            objectness,
            box_regression,
            labels,
            regression_targets,
            self.base_size
        )

        classmap_loss = F.cross_entropy(classmap_logit, classmap.squeeze(-1).long())

        return objectness_loss, box_loss, classmap_loss

    def geo_loss(self, objectness, box_regression, labels, box_target, base_size):
        # objectness: [N, grid sizes of all level, 1]
        # box_regression: [N, grid sizes of all level, (top, right, bottom, left, angle)]
        # labels: same as objectness
        # box_target: same as box_regression

        # gt in arc, pred in [0, 1]
        angle_pred = (box_regression[..., 4:] - 0.5) * 3.14159265358979 / 2.
        angle_gt = box_target[..., 4:5]

        # angle_gt_np = angle_gt.data.cpu().numpy().reshape(-1, 1)
        # label_np = labels.data.cpu().numpy().reshape(-1)
        # print("angle_gt_np:", angle_gt_np.shape, label_np.shape)

        # print("pos_angle:", np.unique(angle_gt_np[label_np == 1]))

        bbox_pred = box_regression[..., :4] * base_size
        bbox_tar = box_target[..., :4]

        loss_objectness = self.dice_loss(labels, objectness)

        loss_box, loss_theta = self.IoU_loss(bbox_tar, bbox_pred, angle_gt, angle_pred, labels)

        return loss_objectness * 0.1, loss_box + 20. * loss_theta

    def dice_loss(self, y_true_cls, y_pred_cls):
        eps = 1e-10
        intersection = (y_true_cls * y_pred_cls).sum()
        union = (y_true_cls).sum() + (y_pred_cls).sum() + eps
        # print("intersection:", intersection, union)
        loss = 1. - (2 * intersection / union)
        return loss

    def IoU_loss(self, gt_geomap, pred_geomap, theta_gt, theta_pred, gt_scoremap):

        d1_gt, d2_gt, d3_gt, d4_gt = torch.split(gt_geomap, split_size_or_sections=1, dim=-1)

        d1_pred, d2_pred, d3_pred, d4_pred = torch.split(pred_geomap, split_size_or_sections=1, dim=-1)

        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)

        area_intersect = (w_union * h_union)
        area_union = (area_gt + area_pred - area_intersect)
        L_AABB = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        L_theta = torch.abs(theta_pred - theta_gt)

        geometry_AABB = (L_AABB * gt_scoremap).mean()
        geometry_theta = (L_theta * gt_scoremap).mean()

        return geometry_AABB, geometry_theta


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    size_stack = cfg.MODEL.ARPN.SIZE_STACK
    scale_stack = cfg.MODEL.ARPN.SCALE_STACK
    base_size = cfg.MODEL.ARPN.BASE_SIZE

    loss_evaluator = RPNLossComputation(matcher,
                                        fg_bg_sampler,
                                        box_coder,
                                        base_size=base_size,
                                        scale_stack=scale_stack,
                                        size_stack=size_stack
                                        )
    return loss_evaluator
