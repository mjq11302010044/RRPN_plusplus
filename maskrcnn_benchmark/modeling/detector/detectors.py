# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .generalized_rrpn_rcnn import GeneralizedRRPNRCNN
from .generalized_arpn_rcnn import GeneralizedARPNRCNN

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN,
                                 "RRPN": GeneralizedRRPNRCNN,
                                 "ARPN": GeneralizedARPNRCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
