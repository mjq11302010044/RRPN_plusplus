# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from maskrcnn_benchmark.modeling.rbox_coder import RBoxCoder
from maskrcnn_benchmark.structures.bounding_box import RBoxList
from maskrcnn_benchmark.structures.rboxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.rboxlist_ops import remove_small_boxes
from maskrcnn_benchmark.structures.rboxlist_ops import eastbox2rbox, set2rboxes
from maskrcnn_benchmark.structures.rboxlist_ops import boxlist_nms, cluster_nms

from ..utils import cat
import numpy as np

class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
        self,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        nms_type="remove",
        box_coder=None,
        fpn_post_nms_top_n=None,
        base_size = 640.,
        scale_stack = [0.25, 0.125, 0.0625, 0.03125, 0.015625],
        score_thresh = 0.1
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(RPNPostProcessor, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        self.base_size = base_size
        self.score_thresh = score_thresh

        if box_coder is None:
            box_coder = RBoxCoder(weights=(1.0, 1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder
        self.scale_stack = scale_stack
        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.nms_fn = boxlist_nms if nms_type == "remove" else cluster_nms

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].bbox.device
        # new_targets = []
        ############ change width & height ############

        new_targets = [target.set2rboxes() for target in targets]

        ###############################################

        gt_boxes = [target.copy_with_fields([]) for target in new_targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))

        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def forward_for_single_feature_map(self, anchors, objectness_, box_regression_, scale):
        """
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 5, H, W

        """
        device = objectness_.device
        N, A, H, W = objectness_.shape

        width, height = anchors[0].size
        # scale = width / W

        # put in the same format as anchors
        objectness = objectness_.permute(0, 2, 3, 1)
        objectness = objectness.reshape(N, -1)
        # get the first 5 channels
        box_regression = box_regression_[:, :5].view(N, -1, 5, H, W).permute(0, 3, 4, 1, 2)
        box_regression = box_regression.reshape(N, -1, 5)

        all_proposals = eastbox2rbox(box_regression, self.base_size, (H, W), scale)

        num_anchors = A * H * W

        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)

        batch_idx = torch.arange(N, device=device)[:, None]
        proposals = all_proposals.view(N, -1, 5)[batch_idx, topk_idx]
        image_shapes = [box.size for box in anchors]

        result = []
        for proposal, score, im_shape in zip(proposals, objectness, image_shapes):

            if not self.training:
                # print("score:", score.shape)
                # print("proposal:", proposal.shape)

                proposal = proposal[score > self.score_thresh]
                score = score[score > self.score_thresh]

                # print("score:", score.shape, score)
                # print("proposal:", proposal.shape)
            # print("score:", score)
            boxlist = RBoxList(proposal, im_shape, mode="xywha")
            boxlist.add_field("objectness", score)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            boxlist = self.nms_fn(
                boxlist,
                self.nms_thresh,
                max_proposals=self.post_nms_top_n,
                score_field="objectness",
            )
            result.append(boxlist)
        return result

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        for a, o, b, s in zip(anchors, objectness, box_regression, self.scale_stack):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b, s))

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)

        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # TODO resolve this difference and make it consistent. It should be per image,
        # and not per batch
        if self.training:
            objectness = torch.cat(
                [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
            )
            box_sizes = [len(boxlist) for boxlist in boxlists]
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


def make_rpn_postprocessor(config, rpn_box_coder, is_train):
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
    if not is_train:
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST

    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST

    min_size = config.MODEL.RPN.MIN_SIZE

    nms_thresh = config.MODEL.ARPN.NMS_THRESH
    nms_type = config.MODEL.ARPN.NMS_TYPE
    scale_stack = config.MODEL.ARPN.SCALE_STACK
    base_size = config.MODEL.ARPN.BASE_SIZE
    score_thresh = config.MODEL.ARPN.SCORE_THRESH

    box_selector = RPNPostProcessor(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        nms_type=nms_type,
        scale_stack=scale_stack,
        base_size=base_size,
        score_thresh=score_thresh
    )
    return box_selector
