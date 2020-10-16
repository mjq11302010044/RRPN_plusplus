# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from maskrcnn_benchmark.structures.bounding_box import RBoxList, BoxList


# TODO check if want to return a single BoxList or a composite
# object
class MaskPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, masker=None):
        super(MaskPostProcessor, self).__init__()
        self.masker = masker

    def forward(self, x, boxes):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        mask_prob = x.sigmoid()

        # select masks coresponding to the predicted classes
        num_masks = x.shape[0]
        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_masks, device=labels.device)
        mask_prob = mask_prob[index, labels][:, None]

        boxes_per_image = [len(box) for box in boxes]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)

        if self.masker:
            mask_prob = self.masker(mask_prob, boxes)

        results = []
        for prob, box in zip(mask_prob, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xywha")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("mask", prob)
            results.append(bbox)

        return results


class RRPNRecProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, masker=None, max_step=35):
        super(RRPNRecProcessor, self).__init__()
        self.masker = masker

    def forward(self, x, boxes):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        # mask_prob = x.sigmoid()

        # [T, B, C] -> [B, T, C]
        word_probs = x.permute(1, 0, 2).softmax(2)

        # select masks coresponding to the predicted classes
        num_words = word_probs.shape[0]
        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        # index = torch.arange(num_words, device=word_probs.device)
        # word_probs = word_probs[index][:, None]

        boxes_per_image = [len(box) for box in boxes]
        word_probs = word_probs.split(boxes_per_image, dim=0)

        results = []
        for prob, box in zip(word_probs, boxes):
            bbox = RBoxList(box.bbox, box.size, mode="xywha")

            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("word_probs", prob)
            results.append(bbox)

        return results


def greedy_decode(model, src, src_mask, max_len=35, start_symbol=1):

    # memory = model.encoder(src, src_mask)

    conv_feature, global_info = model.encoder(src)

    #######################################
    word_probs = []

    batch_bum = src.size()[0]

    ys = torch.ones(batch_bum, 1).fill_(start_symbol).long().cuda()

    # print("conv_feature:", conv_feature.shape)

    for i in range(max_len):

        text = model.embedding_radical(ys)
        blank = model.pe(torch.zeros(text.shape).cuda()).cuda()
        # result = torch.cat([text, blank], 2)
        # batch, seq_len, _ = result.shape
        # print("result:", result.shape, text.shape, blank.shape)
        # print("text:", text.shape, blank.shape, global_info.shape)
        global_info_transfer = (global_info.squeeze(2).squeeze(2))[:, None].repeat(1, text.size(1), 1)
        result = torch.cat([text + blank, global_info_transfer], 2)

        for decoder in model.decoders:
            result = decoder(result, global_info, conv_feature, None)
        # prob: [n, 1, c]
        # print("result:", result.shape)
        prob = model.generator_radical(result[:, -1]).softmax(-1)[:, None]
        word_probs.append(prob)
        # print("prob", prob.shape)
        _, next_word = torch.max(prob[:, 0], dim=-1)
        # next_word = next_word.data[0]
        # print("next_word:", next_word.shape, next_word)
        # word_porb = word_porb.data[0]
        next_word = next_word
        ys = torch.cat([ys,
                        next_word[:, None]], dim=1)
        # print("ys:", ys)
        # if next_word.item() == 0:
        #     break
    # ret = ys.cpu().numpy()[0]
    return torch.cat(word_probs, dim=1)


class RRPNRecTransProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, masker=None, max_step=35):
        super(RRPNRecTransProcessor, self).__init__()
        self.masker = masker
        self.max_step = max_step
        self.src_mask = Variable(torch.from_numpy(np.ones([1, 1, self.max_step], dtype=np.int32)).cuda())

    def forward(self, x, boxes, transformer):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        # mask_prob = x.sigmoid()

        # [B, T, C]
        # word_probs = x.permute(1, 0, 2).softmax(2)

        # select masks coresponding to the predicted classes
        # num_words = word_probs.shape[0]
        # labels = [bbox.get_field("labels") for bbox in boxes]
        # labels = torch.cat(labels)
        # index = torch.arange(num_words, device=word_probs.device)
        # word_probs = word_probs[index][:, None]

        boxes_per_image = [len(box) for box in boxes]
        word_probs = x.split(boxes_per_image, dim=0)

        results = []
        for x_feature, box in zip(word_probs, boxes):
            bbox = RBoxList(box.bbox, box.size, mode="xywha")

            predict_prob = greedy_decode(transformer, x_feature, self.src_mask, self.max_step)

            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("word_probs", predict_prob)
            results.append(bbox)

        return results



class MaskPostProcessorCOCOFormat(MaskPostProcessor):
    """
    From the results of the CNN, post process the results
    so that the masks are pasted in the image, and
    additionally convert the results to COCO format.
    """

    def forward(self, x, boxes):
        import pycocotools.mask as mask_util
        import numpy as np

        results = super(MaskPostProcessorCOCOFormat, self).forward(x, boxes)
        for result in results:
            masks = result.get_field("mask").cpu()
            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")
            result.add_field("mask", rles)
        return results


# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily gor paste_mask_in_image
def expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
    padded_mask, scale = expand_masks(mask[None], padding=padding)
    mask = padded_mask[0, 0]
    box = expand_boxes(box[None], scale)[0]
    box = box.to(dtype=torch.int32)

    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = mask.to(torch.float32)
    mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    if thresh >= 0:
        mask = mask > thresh
    else:
        # for visualization and debugging, we also
        # allow it to return an unmodified mask
        mask = (mask * 255).to(torch.uint8)

    im_mask = torch.zeros((im_h, im_w), dtype=torch.uint8)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
    ]
    return im_mask


class Masker(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold=0.5, padding=1):
        self.threshold = threshold
        self.padding = padding

    def forward_single_image(self, masks, boxes):
        boxes = boxes.convert("xyxy")
        im_w, im_h = boxes.size
        res = [
            paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding)
            for mask, box in zip(masks, boxes.bbox)
        ]
        if len(res) > 0:
            res = torch.stack(res, dim=0)[:, None]
        else:
            res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        return res

    def __call__(self, masks, boxes):
        if isinstance(boxes, BoxList):
            boxes = [boxes]

        # Make some sanity check
        assert len(boxes) == len(masks), "Masks and boxes should have the same length."

        # TODO:  Is this JIT compatible?
        # If not we should make it compatible.
        results = []
        for mask, box in zip(masks, boxes):
            assert mask.shape[0] == len(box), "Number of objects should be the same."
            result = self.forward_single_image(mask, box)
            results.append(result)
        return results


PROCESSERS = {
    "REF_TRANSFORMER": RRPNRecTransProcessor,
    "ORIGINAL": RRPNRecProcessor,
    "REFINED": RRPNRecProcessor,
    "REF_SHORTCUT": RRPNRecProcessor,
}

def make_roi_rec_post_processor(cfg):
    #if cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS:
    #    mask_threshold = cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD
    #    masker = Masker(threshold=mask_threshold, padding=1)
    #else:
    #    masker = None
    #mask_post_processor = MaskPostProcessor(masker)

    rec_post_processor = PROCESSERS[cfg.MODEL.ROI_REC_HEAD.STRUCT](
        max_step=cfg.MODEL.ROI_REC_HEAD.POOLER_RESOLUTION[-1]
    )

    return rec_post_processor
