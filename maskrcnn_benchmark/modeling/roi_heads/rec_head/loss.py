# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import torch
from torch.nn import functional as F
from torch.autograd import Variable

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.rboxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.utils.rec_utils import StrLabelConverter
import numpy as np
_DEBUG = True

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, feature, trg_y, trg, pad=0):
        self.feature = feature
        self.src_mask = Variable(torch.from_numpy(np.ones([feature.size(0), 1, 35], dtype=np.uint8)).cuda())
        if trg is not None:
            self.trg = Variable(trg.cuda(), requires_grad=False)
            self.trg_y = Variable(trg_y.cuda(), requires_grad=False)
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return Variable(tgt_mask.cuda(), requires_grad=False)

def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )
    # TODO put the proposals on the CPU, as the representation for the
    # masks is not efficient GPU-wise (possibly several small tensors for
    # representing a single instance mask)
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.convert(mode="mask")
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class RRPNRecLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='none')
        self.negative_num = 8

        if _DEBUG:
            pro_name = './data_cache/alphabet_IC13_IC15_Syn800K_pro.txt'# './data_cache/alphabet_IC13_IC15_Syn800K_pro.txt'
            self.show_cnt = 0
            if os.path.isfile(pro_name):
                self.alphabet = '-' + open(pro_name, 'r').read()
            else:
                print('Empty alphabet...')
                self.alphabet = '-'
            # self.converter = StrLabelConverter(alphabet)

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "words "fields for creating the targets
        target = target.copy_with_fields(["labels", "words", "word_length"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        words = []
        word_lens = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            words_seq = matched_targets.get_field("words")
            # words_seq = words_seq[positive_inds]
            words_len = matched_targets.get_field("word_length")
            # words_len = words_len[positive_inds]
            # positive_proposals = proposals_per_image[positive_inds]

            # masks_per_image = project_masks_on_boxes(
            #     segmentation_masks, positive_proposals, self.discretization_size
            # )

            labels.append(labels_per_image)
            words.append(words_seq)
            word_lens.append(words_len)

        return labels, words, word_lens

    def __call__(self, proposals, word_logits, targets):
        labels, words, word_lens = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        word_targets = cat(words, dim=0)
        word_lens = cat(word_lens, dim=0)

        ########################## positive samples ###########################
        positive_inds = torch.nonzero(labels > 0).squeeze(1)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if word_targets.numel() == 0:
            return word_logits.sum() * 0
        pos_logits = word_logits[:, positive_inds] .log_softmax(2)
        pos_wlens = word_lens[positive_inds]
        pos_target = word_targets[positive_inds]
        # print("word_lens:", word_lens, positive_inds)
        limited_ind = pos_wlens < 18
        word_lens_lim = pos_wlens[limited_ind]
        word_targets_lim = pos_target[limited_ind]
        pos_logits_lim = pos_logits[:, limited_ind]

        if word_targets_lim.numel() == 0:
            return pos_logits.sum() * 0

        batch_size = pos_logits_lim.size()[1]
        predicted_length = torch.tensor([pos_logits_lim.size(0)] * batch_size)

        # print('words_targets:', word_targets)
        word_targets_flatten = word_targets_lim.view(-1)
        positive_w_inds = torch.nonzero(word_targets_flatten > 0).squeeze(1)
        # print('positive_inds:', positive_inds)
        word_targets_flatten = word_targets_flatten[positive_w_inds]


        if _DEBUG:
            self.show_cnt += 1
            if  self.show_cnt % 100 == 0:
                pos_logits_show = pos_logits_lim.permute(1, 0, 2)
                pos_value, pos_inds = pos_logits_show.max(2)
                # print('word_lens_lim:', word_lens_lim)
                # print('pos_logits:', pos_inds, word_targets_flatten)
                predict_seq = pos_inds.data.cpu().numpy()
                word_targets_np = word_targets_lim.data.cpu().numpy()
                for a in range(predict_seq.shape[0]):
                    pred_str = ''
                    gt_str = ''
                    for b in range(predict_seq.shape[1]):
                        pred_str += self.alphabet[predict_seq[a, b]]
                    for c in range(word_targets_np.shape[1]):
                        if word_targets_np[a, c] != 0:
                            #print('use int?', word_targets_np[a, c])
                            gt_str += self.alphabet[int(word_targets_np[a, c])]
                    # print('lstr:', pred_str, gt_str)
                    print('lstr:', "|" + pred_str + "|", "|" + gt_str + "|")

        return self.ctc_loss(
            pos_logits_lim,
            word_targets_flatten.long(),
            predicted_length.long(),
            word_lens_lim.long()
        ).sum() / pos_logits.size()[0] / batch_size


class RRPNRecLossBalancedComputation(object):
    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='none')
        self.negative_num = 8

        if _DEBUG:
            pro_name = './data_cache/alphabet_IC13_IC15_Syn800K_pro.txt' # alphabet_90Klex_IC13_IC15_Syn800K_pro.txt
            self.show_cnt = 0
            if os.path.isfile(pro_name):
                self.alphabet = '-' + open(pro_name, 'r').read()
            else:
                print('Empty alphabet...')
                self.alphabet = '-'
            # self.converter = StrLabelConverter(alphabet)

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "words "fields for creating the targets
        target = target.copy_with_fields(["labels", "words", "word_length"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        words = []
        word_lens = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            words_seq = matched_targets.get_field("words")
            # words_seq = words_seq[positive_inds]
            words_len = matched_targets.get_field("word_length")
            # words_len = words_len[positive_inds]
            # positive_proposals = proposals_per_image[positive_inds]

            # masks_per_image = project_masks_on_boxes(
            #     segmentation_masks, positive_proposals, self.discretization_size
            # )

            labels.append(labels_per_image)
            words.append(words_seq)
            word_lens.append(words_len)

        return labels, words, word_lens

    def __call__(self, proposals, word_logits, targets):
        labels, words, word_lens = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        word_targets = cat(words, dim=0)
        word_lens = cat(word_lens, dim=0)

        ########################## positive samples ###########################
        positive_inds = torch.nonzero(labels > 0).squeeze(1)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if word_targets.numel() == 0:
            return word_logits.sum() * 0
        pos_logits = word_logits[:, positive_inds].log_softmax(2)
        pos_wlens = word_lens[positive_inds]
        pos_target = word_targets[positive_inds]

        limited_ind = pos_wlens < 20
        word_lens_lim = pos_wlens[limited_ind]
        word_targets_lim = pos_target[limited_ind]
        pos_logits_lim = pos_logits[:, limited_ind]

        if word_targets_lim.numel() == 0:
            return pos_logits.sum() * 0

        batch_size = pos_logits_lim.size()[1]
        predicted_length = torch.tensor([pos_logits_lim.size(0)] * batch_size)

        # print('words_targets:', word_targets)
        word_targets_flatten = word_targets_lim.view(-1)
        positive_w_inds = torch.nonzero(word_targets_flatten > 0).squeeze(1)
        # print('positive_inds:', positive_inds)
        word_targets_flatten = word_targets_flatten[positive_w_inds]

        ########################## negative samples ###########################

        # picked limited negative samples

        # neg_sample = min(int(batch_size / 8 + 1), self.negative_num)

        negative_inds = torch.nonzero(labels == 0).squeeze(1)[-int(batch_size / 8 + 1):] #
        neg_logits = word_logits[:, negative_inds].log_softmax(2)

        device = neg_logits.device

        batch_size = neg_logits.size()[1]
        neg_predicted_length = torch.tensor([neg_logits.size(0)] * batch_size)

        neg_wlens = torch.tensor([1] * batch_size)
        neg_target = torch.tensor([1] * batch_size).double()

        # Conbining samples
        pos_logits_lim = torch.cat([pos_logits_lim, neg_logits], dim=1)
        word_targets_flatten = torch.cat([word_targets_flatten, neg_target.to(device)], dim=0)
        predicted_length = torch.cat([predicted_length.long(), neg_predicted_length.long()], dim=0)
        word_lens_lim = torch.cat([word_lens_lim, neg_wlens.to(device)], dim=0)

        # print('word_logits:', word_logits.size())
        # print('words_targets_flatten__________b:', b)
        # print('pos_logits:', pos_logits.size())

        if _DEBUG:
            self.show_cnt += 1
            if  self.show_cnt % 100 == 0:

                pos_logits_show = pos_logits_lim.permute(1, 0, 2)
                pos_value, pos_inds = pos_logits_show.max(2)
                # print('word_lens_lim:', word_lens_lim)
                # print('pos_logits:', pos_inds, word_targets_flatten)
                predict_seq = pos_inds.data.cpu().numpy()
                word_targets_np = word_targets_lim.data.cpu().numpy()

                neg_target_np = neg_target.data.cpu().numpy().reshape(-1, 1)
                neg_target_np = np.tile(neg_target_np, (1, word_targets_np.shape[-1]))

                word_targets_np = np.concatenate([word_targets_np, neg_target_np], axis=0)

                for a in range(predict_seq.shape[0]):
                    pred_str = ''
                    gt_str = ''
                    for b in range(predict_seq.shape[1]):
                        pred_str += self.alphabet[predict_seq[a, b]]
                    for c in range(word_targets_np.shape[1]):
                        if word_targets_np[a, c] != 0:
                            #print('use int?', word_targets_np[a, c])
                            gt_str += self.alphabet[int(word_targets_np[a, c])]
                    print('lstr:', "|" + pred_str + "|", "|" + gt_str + "|")

        batch_size = pos_logits_lim.size()[1]
        time_stamp = pos_logits_lim.size()[0]

        ctc_loss = self.ctc_loss(
            pos_logits_lim,
            word_targets_flatten.long(),
            predicted_length.long(),
            word_lens_lim.long()
        )

        # print("ctc_loss:", ctc_loss.shape, ctc_loss)

        ctc_loss = ctc_loss.sum() / float(time_stamp * batch_size + 1e-10)
        return ctc_loss * (ctc_loss > 0.) + 1e-10


class RRPNRecTransLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='none')
        self.negative_num = 8

        if _DEBUG:
            pro_name = './data_cache/alphabet_IC13_IC15_Syn800K_pro.txt'# './data_cache/alphabet_IC13_IC15_Syn800K_pro.txt' # alphabet_90Klex_IC13_IC15_Syn800K_pro.txt
            self.show_cnt = 0
            if os.path.isfile(pro_name):
                self.alphabet = '_' + open(pro_name, 'r').read()
            else:
                print('Empty alphabet...')
                self.alphabet = '_'
            # self.converter = StrLabelConverter(alphabet)

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "words "fields for creating the targets
        target = target.copy_with_fields(["labels", "words", "word_length"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        words = []
        word_lens = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            words_seq = matched_targets.get_field("words")
            words_len = matched_targets.get_field("word_length")

            labels.append(labels_per_image)
            words.append(words_seq)
            word_lens.append(words_len)

        return labels, words, word_lens

    def __call__(self, proposals, word_logits, targets, transformer):
        labels, words, word_lens = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        word_targets = cat(words, dim=0)
        word_lens = cat(word_lens, dim=0)

        ########################## positive samples ###########################
        positive_inds = torch.nonzero(labels > 0).squeeze(1)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if word_targets.numel() == 0:
            return word_logits.sum() * 0

        # word_logits: (n, t, c)
        pos_logits = word_logits[positive_inds]
        pos_wlens = word_lens[positive_inds]
        pos_target = word_targets[positive_inds]

        limited_ind = pos_wlens < 30
        word_lens_lim = pos_wlens[limited_ind]
        word_targets_lim = pos_target[limited_ind]
        pos_logits_lim = pos_logits[limited_ind]

        if word_targets_lim.numel() == 0:
            return pos_logits.sum() * 0

        batch_size = pos_logits_lim.size()[0]

        start_label = 1

        # label: [n, t] word_targets_lim
        labels = torch.zeros((batch_size, pos_logits_lim.size(-1)))\
            .to(pos_logits_lim.device)\
            .type_as(word_targets_lim)
        # print("labels:", labels.shape, word_targets_lim.shape)

        max_len = min(word_targets_lim.size(-1), pos_logits_lim.size(-1))
        labels[:, :max_len] = word_targets_lim.to(pos_logits_lim.device)[:, :max_len]

        labels_rshift = torch.zeros((batch_size, pos_logits_lim.size(-1)))\
            .to(pos_logits_lim.device)\
            .type_as(labels)

        labels_rshift[:, 0] = start_label
        labels_rshift[:, 1:] = labels[:, :-1]

        # print("labels_lshift:", labels_lshift, labels)

        assert labels_rshift.shape == labels.shape, str(labels_rshift.shape) + " != " + str(labels.shape)

        # batch = Batch(pos_logits_lim, labels, labels_lshift)

        # print("pos_logits_lim:", pos_logits_lim.shape)

        # word_lens_lim = word_lens_lim

        trans_out = transformer(pos_logits_lim, word_lens_lim + 1, (labels_rshift).long())

        # print("trans_out:", trans_out.shape, word_targets_lim.shape, labels.shape)
        # print(pos_logits_lim.contiguous().view(-1, pos_logits_lim.size(-1)).shape,
        #       batch.trg.contiguous().view(-1).shape)

        if _DEBUG:
            self.show_cnt += 1
            if  self.show_cnt % 100 == 0:
                # pos_logits_show = pos_logits_lim.permute(1, 0, 2)
                pos_value, pos_inds = trans_out.softmax(2).max(2)
                # print('word_lens_lim:', word_lens_lim)
                # print('pos_logits:', pos_inds, word_targets_flatten)
                predict_seq = pos_inds.data.cpu().numpy()
                word_targets_np = word_targets_lim.data.cpu().numpy()
                for a in range(predict_seq.shape[0]):
                    pred_str = ''
                    gt_str = ''
                    for b in range(predict_seq.shape[1]):
                        pred_str += self.alphabet[predict_seq[a, b]]
                    for c in range(word_targets_np.shape[1]):
                        if word_targets_np[a, c] != 0:
                            #print('use int?', word_targets_np[a, c])
                            gt_str += self.alphabet[int(word_targets_np[a, c])]
                    # print('lstr:', pred_str, gt_str)
                    print('lstr:', "|" + pred_str[:35] + "|", "|" + gt_str + "|")

        # print("trans_out:", trans_out.shape, word_targets_lim.shape, batch.trg_y.shape, word_lens_lim)

        max_step = trans_out.size(1)
        range_tensor = torch.arange(max_step).unsqueeze(0).to(trans_out.device)
        range_tensor = range_tensor.expand(trans_out.size(0), range_tensor.size(1))

        # We need one more ending sign
        label_mask = range_tensor <= word_lens_lim[:, None] + 1

        # print("label_mask:", label_mask.shape, word_lens_lim[:, None] + 1)
        # label_mask: [n, t] -> [n * t]
        label_mask = label_mask.view(-1)
        trans_out = trans_out.contiguous().view(-1, trans_out.size(-1))
        target_y = labels.contiguous().view(-1)

        trans_out = trans_out[label_mask]
        target_y = target_y[label_mask]

        # print("trans_out:", trans_out.shape, target_y.shape, target_y)

        return F.cross_entropy(
            trans_out, target_y.long()
        )#  / (float(word_lens_lim.size(0)) + 1e-10)


class MaskRCNNLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            segmentation_masks = matched_targets.get_field("masks")
            segmentation_masks = segmentation_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            masks_per_image = project_masks_on_boxes(
                segmentation_masks, positive_proposals, self.discretization_size
            )

            labels.append(labels_per_image)
            masks.append(masks_per_image)

        return labels, masks

    def __call__(self, proposals, mask_logits, targets):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        labels, mask_targets = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[positive_inds, labels_pos], mask_targets
        )
        return mask_loss


LOSS_TYPE = {
    "ORIGINAL": RRPNRecLossComputation,
    "REFINED": RRPNRecLossComputation,
    "REF_SHORTCUT": RRPNRecLossComputation,
    "REF_TRANSFORMER": RRPNRecTransLossComputation
}



def make_roi_rec_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_REC_HEAD.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_REC_HEAD.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = LOSS_TYPE[cfg.MODEL.ROI_REC_HEAD.STRUCT](
            matcher, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
        ) \
    if cfg.MODEL.ROI_REC_HEAD.POS_ONLY else \
        RRPNRecLossBalancedComputation(
            matcher, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
    )

    return loss_evaluator
