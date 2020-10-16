# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
import numpy as np
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.rbox_coder import RBoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.rboxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.utils.visualize import vis_image
from PIL import Image

_DEBUG = False

class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder, edge_punished=False, OHEM=False, angle_thres=15.):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.edge_punished = edge_punished
        self.OHEM = OHEM
        self.angle_thres = angle_thres

        self.iter_cnt = 0

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # print('matched_idxs', matched_idxs)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]

        ############### ANGLE MATCHER HERE ###############

        A_proposal = proposal.bbox[:, -1]
        A_target = matched_targets.bbox[:, -1]

        # [N_anc, N_tar]
        angle_diff_matched = torch.abs(A_proposal - A_target)
        angle_filter = angle_diff_matched > self.angle_thres

        matched_idxs[angle_filter] = Matcher.BELOW_LOW_THRESHOLD

        ##################################################

        # updating targets
        matched_targets = target[matched_idxs.clamp(min=0)]

        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)
            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            if _DEBUG:
                self.iter_cnt += 1
                if self.iter_cnt % 10 == 0:
                    label_np = labels_per_image.data.cpu().numpy()
                    # print('label shape:', label_np.shape)
                    # print('labels pos/neg:', len(np.where(label_np == 1)[0]), '/', len(np.where(label_np == 0)[0]))
                    imw, imh = proposals_per_image.size
                    proposals_np = proposals_per_image.bbox.data.cpu().numpy()
                    canvas = np.zeros((imh, imw, 3), np.uint8)

                    # pick pos proposals for visualization
                    pos_proposals = proposals_np[label_np == 1]
                    # print('proposals_np:', pos_proposals)
                    pilcanvas = vis_image(Image.fromarray(canvas), pos_proposals, [i for i in range(pos_proposals.shape[0])])
                    pilcanvas.save('proposals_for_rcnn_maskboxes.jpg', 'jpeg')

            # print('proposal target', regression_targets_per_image, np.unique(labels_per_image.data.cpu().numpy()))
            # print('labels_per_image:', labels_per_image.size(), np.unique(label_np))

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        # print('targets:', targets[0].bbox)
        labels, regression_targets = self.prepare_targets(proposals, targets)
        # print('regression_targets:', targets[0].bbox)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
            labels, regression_targets, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)# .detach()
        box_regression = cat(box_regression, dim=0)#.detach()
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )
        if _DEBUG:
            # print('labels:', labels)
            # print('rrpn_labels:', np.unique(labels.data.cpu().numpy()))
            prob = torch.nn.functional.softmax(class_logits, -1)
            # print('probs:', np.unique(prob[:, 1].data.cpu().numpy())[-10:])
            pass
        # print('loss_class_logits:', class_logits)



        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        # pick the target of correct position
        map_inds = 5 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3, 4], device=device)

        box_regression_pos = box_regression[sampled_pos_inds_subset[:, None], map_inds]
        regression_targets_pos = regression_targets[sampled_pos_inds_subset]

        if self.edge_punished:
            proposals_cat = torch.cat([proposal.bbox for proposal in proposals], 0)
            proposals_cat_w = proposals_cat[:, 2:3][sampled_pos_inds_subset]
            proposals_cat_w_norm = proposals_cat_w / (torch.mean(proposals_cat_w) + 1e-10)
            box_regression_pos = proposals_cat_w_norm * box_regression_pos
            regression_targets_pos = proposals_cat_w_norm * regression_targets_pos

        if _DEBUG:
            # print('map_inds:', box_regression[sampled_pos_inds_subset[:, None], map_inds], regression_targets[sampled_pos_inds_subset])
            pass

        if self.OHEM:

            cls_logits = class_logits  # objectness[sampled_inds]
            score_sig = torch.nn.functional.softmax(cls_logits, -1)
            # map_inds = labels_pos

            # max_scores, max_inds = torch.max(score_sig, 1)

            # pick hard positive which takes 1/4
            pos_score_sig = score_sig[sampled_pos_inds_subset, labels_pos]
            # print("pos_score_sig:", pos_score_sig.shape, labels_pos.shape, pos_score_sig)
            pos_num = pos_score_sig.shape[0]
            hard_pos_num  = int(pos_num / 2) + 1
            hp_vals, hp_indices = torch.topk(-pos_score_sig, hard_pos_num, dim=0)
            hard_pos_sig = pos_score_sig[hp_indices]

            # print("hard_pos_sig:", hard_pos_sig, pos_score_sig)

            pos_label = labels_pos
            pos_label = pos_label[hp_indices]
            pos_logits = cls_logits[sampled_pos_inds_subset]
            pos_logits = pos_logits[hp_indices]

            pos_box_reg = box_regression_pos[hp_indices]
            pos_box_target = regression_targets_pos[hp_indices]

            # pick hard negative which takes 1/4
            sampled_neg_inds_subset = torch.nonzero(labels < 1).squeeze(1)
            labels_neg = labels[sampled_neg_inds_subset]

            neg_score_sig = score_sig[sampled_neg_inds_subset, labels_neg]
            # print("neg_score_sig:", neg_score_sig.shape, labels_neg.shape, neg_score_sig)
            neg_num = neg_score_sig.shape[0]
            hard_neg_num = int(neg_num / 2) + 1

            # get top least bg cls scores
            hn_vals, hn_indices = torch.topk(-neg_score_sig, hard_neg_num, dim=0)
            hard_neg_sig = neg_score_sig[hn_indices]

            # print("hard_neg_sig:", hard_neg_sig, neg_score_sig)

            neg_label = labels_neg
            neg_label = neg_label[hn_indices]
            neg_logits = cls_logits[sampled_neg_inds_subset]
            neg_logits = neg_logits[hn_indices]

            hard_labels = torch.cat([pos_label, neg_label], dim=0)
            hard_logits = torch.cat([pos_logits, neg_logits], dim=0)

            ohem_box_loss = smooth_l1_loss(
                pos_box_reg,
                pos_box_target,
                beta=1.0 / 9,
                size_average=False,
            ) / float(hard_pos_num + hard_neg_num)

            ohem_objectness_loss = F.cross_entropy(
                hard_logits, hard_labels.to(hard_logits.device)
            )

            return ohem_objectness_loss, ohem_box_loss

        else:
            classification_loss = F.cross_entropy(class_logits, labels)

            box_loss = smooth_l1_loss(
                box_regression_pos,
                regression_targets_pos,
                size_average=False,
                beta=1,
            )

            box_loss = box_loss / (labels.numel() + 1e-10)

            return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.RBBOX_REG_WEIGHTS
    box_coder = RBoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    edge_punished = cfg.MODEL.EDGE_PUNISHED
    loss_evaluator = FastRCNNLossComputation(matcher, fg_bg_sampler, box_coder, edge_punished)

    return loss_evaluator
