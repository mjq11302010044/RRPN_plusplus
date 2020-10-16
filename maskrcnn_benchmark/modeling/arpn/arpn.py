# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.rbox_coder import RBoxCoder
from maskrcnn_benchmark.layers import Mish
from .loss import make_rpn_loss_evaluator
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor
from .mc_loss import make_rpn_loss_evaluator as make_mc_rpn_loss_evaluator

loss_evaluator_dict = {
    "SingleConvARPNHead": make_rpn_loss_evaluator,
    "SingleConvARPNMCHead": make_mc_rpn_loss_evaluator,
    "TowerARPNHead": make_rpn_loss_evaluator
}

class Conv2dGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, gn=False):
        super(Conv2dGroup, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        # self.bn = nn.BatchNorm2d(out_channels) if bn else None #
        self.gn = nn.GroupNorm(32, out_channels) if gn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


@registry.RPN_HEADS.register("SingleConvARPNHead")
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors=1):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted

        # We consider the condition in east
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
             in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        self.cls_logits = nn.Conv2d(in_channels, 1 * num_anchors, kernel_size=1, stride=1)

        # Dis from t, r, b, l
        self.bbox_pred = nn.Conv2d(
            in_channels, 4 * num_anchors, kernel_size=1, stride=1
        )

        # Angle
        self.angle_pred = nn.Conv2d(in_channels, 1 * num_anchors, kernel_size=1, stride=1)

        for l in [self.conv, self.cls_logits, self.bbox_pred, self.angle_pred]: #
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        self.activation = Mish() if cfg.MODEL.ARPN.USE_MISH else nn.ReLU()

    def forward(self, x):
        logits = []
        bbox_reg = []
        # angle_pred = []

        for feature in x:
            t = self.activation(self.conv(feature))

            logits.append(self.cls_logits(t).sigmoid())
            bbox_logit = torch.cat([
                self.bbox_pred(t).sigmoid(),
                self.angle_pred(t).sigmoid()
            ], dim=1)

            bbox_reg.append(bbox_logit)

        return logits, bbox_reg # , angle_pred


@registry.RPN_HEADS.register("SingleConvARPNMCHead")
class MCRPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors=1):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted

        # We consider the condition in east
        """
        super(MCRPNHead, self).__init__()
        self.conv = nn.Conv2d(
             in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        self.cls_logits = nn.Conv2d(in_channels, 1 * num_anchors, kernel_size=1, stride=1)

        # Dis from t, r, b, l
        self.bbox_pred = nn.Conv2d(
            in_channels, 4 * num_anchors, kernel_size=1, stride=1
        )

        # Angle
        self.angle_pred = nn.Conv2d(in_channels, 1 * num_anchors, kernel_size=1, stride=1)

        # Multi-class map
        self.mc_pred = nn.Conv2d(in_channels, cfg.MODEL.ARPN.MC_NUM * num_anchors, kernel_size=1, stride=1)

        for l in [self.conv, self.cls_logits, self.bbox_pred, self.angle_pred, self.mc_pred]: #
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        # angle_pred = []

        for feature in x:
            t = F.relu(self.conv(feature))

            logits.append(self.cls_logits(t).sigmoid())
            bbox_logit = torch.cat([
                self.bbox_pred(t).sigmoid(),
                self.angle_pred(t).sigmoid(),
                self.mc_pred(t),
            ], dim=1)

            bbox_reg.append(bbox_logit)

        return logits, bbox_reg # , angle_pred


@registry.RPN_HEADS.register("TowerARPNHead")
class TowerARPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors=1):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted

        # We consider the condition in east
        """
        super(TowerARPNHead, self).__init__()
        self.conv = nn.Conv2d(
             in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        cls_tower = []
        abox_tower = []

        for i in range(cfg.MODEL.ARPN.CONV_STACK):
            #cls_tower.append(
            #    Conv2dGroup(
            #        in_channels,
            #        in_channels,
            #        3,
            #        same_padding=True,
            #        gn=cfg.MODEL.ARPN.USE_GN
            #    )
            #)

            abox_tower.append(
                Conv2dGroup(
                    in_channels,
                    in_channels,
                    3,
                    same_padding=True,
                    gn=cfg.MODEL.ARPN.USE_GN
                )
            )

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('abox_tower', nn.Sequential(*abox_tower))

        self.cls_logits = nn.Conv2d(in_channels, 1 * num_anchors, kernel_size=1, stride=1)

        # Dis from t, r, b, l
        self.bbox_pred = nn.Conv2d(
            in_channels, 4 * num_anchors, kernel_size=1, stride=1
        )

        # for i in range(len(cfg.MODEL.ARPN.SCALE_STACK)):
        #     self.add_module('ff_boxes_' + str(i + 2), self.box_logits[i])
        # Angle
        self.angle_pred = nn.Conv2d(in_channels, 1 * num_anchors, kernel_size=1, stride=1)

        # initialization
        for modules in [self.cls_tower, self.abox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.angle_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        '''
        for name, param in self.named_parameters():
            # print('name:', name)
            if "weight" in name and 'gn' in name:
                param.data.fill_(1)
            elif "bias" in name and 'gn' in name:
                param.data.fill_(0)
            else:
                torch.nn.init.normal_(param, std=0.01)
                torch.nn.init.constant_(param, 0)
        
        for l in [self.conv, self.cls_logits, self.angle_pred] + self.box_logits: # self.bbox_pred,
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)
        '''
    def forward(self, x):
        logits = []
        bbox_reg = []
        # angle_pred = []

        cnt = 0

        for feature in x:

            # cls = self.cls_tower(feature)
            t = self.abox_tower(feature)

            logits.append(self.cls_logits(t).sigmoid())
            bbox_logit = torch.cat([
                self.bbox_pred(t).sigmoid(), # .sigmoid(),
                self.angle_pred(t).sigmoid()
            ], dim=1)

            bbox_reg.append(bbox_logit)

            cnt += 1

        return logits, bbox_reg


loss_name_dict = {
    "SingleConvARPNHead": ["loss_objectness", "loss_rpn_box_reg"],
    "SingleConvARPNMCHead": ["loss_objectness", "loss_rpn_box_reg", "loss_mc"],
    "TowerARPNHead": ["loss_objectness", "loss_rpn_box_reg"],
}


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator(cfg)

        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        head = rpn_head(
            cfg, in_channels, 1
        )

        rpn_box_coder = RBoxCoder(weights=(1.0, 1.0, 1.0, 1.0, 1.0))

        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        loss_evaluator = loss_evaluator_dict[cfg.MODEL.RPN.RPN_HEAD](cfg, rpn_box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.loss_name = loss_name_dict[cfg.MODEL.RPN.RPN_HEAD]

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets
                )
        loss_item = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )

        losses = {self.loss_name[i]: loss_item[i] for i in range(len(loss_item))}

        '''
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        '''
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


def build_rpn(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return RPNModule(cfg)
