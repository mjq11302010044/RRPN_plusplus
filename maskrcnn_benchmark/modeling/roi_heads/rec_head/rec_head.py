# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList, RBoxList
from maskrcnn_benchmark.structures.rboxlist_ops import cat_boxlist
from maskrcnn_benchmark.layers import Transformer

from .roi_rec_feature_extractors import make_roi_rec_feature_extractor
from .roi_rec_predictors import make_roi_rec_predictor
from .inference import make_roi_rec_post_processor
from .loss import make_roi_rec_loss_evaluator



def keep_only_positive_boxes(boxes, max_num=128):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], RBoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds][:max_num])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


def sampling_boxes(boxes, max_num=128):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], RBoxList)
    assert boxes[0].has_field("labels")

    all_boxes = []

    positive_boxes = []
    positive_inds = []

    negative_boxes = []
    negative_inds = []

    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds][:max_num])
        positive_inds.append(inds_mask)

        neg_mask = labels == 0
        neg_inds = neg_mask.nonzero().squeeze(1)

        negative_box = boxes_per_image[neg_inds][-int(max_num / 4):]

        negative_boxes.append(negative_box)
        negative_inds.append(neg_mask)

        all_boxes.append(cat_boxlist([boxes_per_image[inds][:max_num], negative_box]))

    return positive_boxes, positive_inds, negative_boxes, negative_inds, all_boxes


class ROIRecHead(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIRecHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_rec_feature_extractor(cfg)
        self.predictor = make_roi_rec_predictor(cfg)
        self.post_processor = make_roi_rec_post_processor(cfg)
        self.loss_evaluator = make_roi_rec_loss_evaluator(cfg)

        self.max_num_positive = self.cfg.MODEL.ROI_REC_HEAD.MAX_POSITIVE_NUM

        if self.cfg.MODEL.ROI_REC_HEAD.STRUCT == "REF_TRANSFORMER":
            al_profile = cfg.MODEL.ROI_REC_HEAD.ALPHABET

            if os.path.isfile(al_profile):
                num_classes = len(open(al_profile, 'r').read()) + 1
            else:
                print("We don't expect you to use default class number...Retry it")
                num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

            self.transformer = Transformer(self.cfg, num_classes)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            positive_boxes, positive_inds, negative_boxes, negative_inds, proposals = sampling_boxes(proposals, self.max_num_positive)
            if self.cfg.MODEL.ROI_REC_HEAD.POS_ONLY:
                proposals = positive_boxes


        if self.training and self.cfg.MODEL.ROI_REC_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            pos_x = x[torch.cat(positive_inds, dim=0)][:self.max_num_positive]

            all_proposals = cat_boxlist(all_proposals)
            pos_proposals = all_proposals[torch.cat(positive_inds, dim=0)][:self.max_num_positive]

            if self.cfg.MODEL.ROI_REC_HEAD.POS_ONLY:
                x = pos_x
                proposals = pos_proposals
            else:
                neg_x = x[torch.cat(negative_inds, dim=0)][:self.max_num_positive]
                x = torch.cat([pos_x, neg_x], dim=0)
                neg_proposals = all_proposals[torch.cat(negative_inds, dim=0)][:self.max_num_positive]
                proposals = cat_boxlist([pos_proposals, neg_proposals])

        else:

            if not self.training:
                proposals = [proposal.rescale(self.cfg.MODEL.ROI_REC_HEAD.BOXES_MARGIN)
                             for proposal in proposals]

            x = self.feature_extractor(features, proposals)

            if self.training and self.cfg.MODEL.ROI_REC_HEAD.REC_DETACH:
                x = x.detach()

        rec_logits = self.predictor(x)

        if not self.training:
            if self.cfg.MODEL.ROI_REC_HEAD.STRUCT == "REF_TRANSFORMER":
                result = self.post_processor(rec_logits, proposals, self.transformer)
            else:
                result = self.post_processor(rec_logits, proposals)
            return x, result, {}

        if self.cfg.MODEL.ROI_REC_HEAD.STRUCT == "REF_TRANSFORMER":
            loss_rec = self.loss_evaluator(proposals, rec_logits, targets, self.transformer)
        else:
            loss_rec = self.loss_evaluator(proposals, rec_logits, targets)

        return x, proposals, dict(loss_rec=loss_rec)


def build_roi_rec_head(cfg):
    return ROIRecHead(cfg)
