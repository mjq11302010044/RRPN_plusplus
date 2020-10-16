// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "RROIAlign.h"
#include "ROIAlignRotated.h"
#include "ROIAlignRotatedKeep.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");

  m.def("rroi_align_forward", &RROIAlign_forward, "RROIAlign_forward");
  m.def("rroi_align_backward", &RROIAlign_backward, "RROIAlign_backward");

  m.def(
      "roi_align_rotated_forward",
      &ROIAlignRotated_forward,
      "Forward pass for Rotated ROI-Align Operator");
  m.def(
      "roi_align_rotated_backward",
      &ROIAlignRotated_backward,
      "Backward pass for Rotated ROI-Align Operator");

  m.def(
      "roi_align_rotated_keep_forward",
      &ROIAlignRotatedKeep_forward,
      "Forward pass for Rotated-Keep ROI-Align Operator");
  m.def(
      "roi_align_rotated_keep_backward",
      &ROIAlignRotatedKeep_backward,
      "Backward pass for Rotated-Keep ROI-Align Operator");
}
