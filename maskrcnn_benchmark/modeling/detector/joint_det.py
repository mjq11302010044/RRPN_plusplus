# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
# AFPN components
from ..arpn.arpn import build_rpn as build_arpn
from ..roi_heads.rroi_heads import build_roi_heads as build_rroi_heads
# Faster-RCNN components
from ..rpn.rpn import build_rpn as build_general_rpn
from ..roi_heads.roi_heads import build_roi_heads

class JointDET(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(JointDET, self).__init__()

        self.backbone = build_backbone(cfg)

        self.arpn = build_arpn(cfg)
        self.rroi_heads = build_rroi_heads(cfg)

        self.general_rpn = build_general_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)

        self.fp4p_on = cfg.MODEL.FP4P_ON

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)

        # Targets: List[[RBoxList, BoxList]] contain all boxes from corresponding image batches
        # We need to separate from each
        text_targets = None
        gen_targets = None
        if self.training and targets is None:
            text_targets = []
            gen_targets = []
            for tar in targets:
                txt_tar = tar["text"]
                gen_tar = tar["general"]
                text_targets.append(txt_tar)
                gen_targets.append(gen_tar)

        text_proposals, txt_proposal_losses = self.arpn(images, features, text_targets)
        gen_proposals, gen_proposal_losses = self.general_rpn(images, features, gen_targets)

        # features = [feature.detach() for feature in features]

        if self.roi_heads:
            if self.training:
                # Change target to rboxes
                text_targets = [text_target.set2rboxes() for text_target in text_targets]

            txt_x, txt_result, txt_detector_losses = self.rroi_heads(features, text_proposals, text_targets)
            gen_x, gen_result, gen_detector_losses = self.roi_heads(features, gen_proposals, gen_targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            txt_result = text_proposals
            gen_result = gen_proposals
            detector_losses = {}

        if self.training:
            losses = {}

            losses.update({"txt_" + k: txt_proposal_losses[k] for k in txt_proposal_losses})
            losses.update({"gen_" + k: gen_proposal_losses[k] for k in gen_proposal_losses})
            losses.update({"txt_" + k: txt_detector_losses[k] for k in txt_detector_losses})
            losses.update({"gen_" + k: gen_detector_losses[k] for k in gen_detector_losses})

            return losses

        return [txt_result, gen_result]
