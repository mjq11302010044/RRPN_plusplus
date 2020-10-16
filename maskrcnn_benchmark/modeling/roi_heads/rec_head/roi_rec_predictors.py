# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F
import os
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.layers import Mish
from maskrcnn_benchmark.modeling.make_layers import group_norm as GN

class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=None):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = relu if not relu is None else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv2dGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=None, same_padding=False, bn=False):
        super(Conv2dGroup, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None #
        self.relu = relu if not relu is None else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RECG(nn.Module):
    def __init__(self, char_class, g_feat_channel=1024, inter_channel=256, bn=True, relu_type="Mish"):
        super(RECG, self).__init__()

        #self.rec_conv1 = nn.Sequential(Conv2dGroup(g_feat_channel, inter_channel, 3, same_padding=True, bn=bn),
        #                              Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn),
        #                              nn.Conv2d(inter_channel, inter_channel, 3, (2, 1), 1))

        # inter_channel *= 2

        activation = Mish() if relu_type == "Mish" else nn.ReLU(inplace=True)

        self.rec_conv1_1 = Conv2dGroup(g_feat_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_conv1_2 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_pool1_1 = nn.Conv2d(inter_channel, inter_channel, 3, (2, 1), 1)

        # self.max_pool1_1 = nn.MaxPool2d((2, 1), stride=(2, 1))

        # self.rec_conv1 = nn.Conv2d(g_feat_channel, inter_channel, 3, stride=1, padding=1)

        # inter_channel *= 2

        #self.rec_conv2 = nn.Sequential(Conv2dGroup(inter_channel // 2, inter_channel, 3, same_padding=True, bn=bn),
        #                                Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn),
        #                                nn.Conv2d(inter_channel, inter_channel, 3, (2, 1), 1))

        self.rec_conv2_1 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_conv2_2 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_pool2_1 = nn.Conv2d(inter_channel, inter_channel, 3, (2, 1), 1)

        # self.max_pool2_1 = nn.MaxPool2d((2, 1), stride=(2, 1))

        # inter_channel *= 2

        # self.rec_conv3 = nn.Sequential(Conv2dGroup(inter_channel // 2, inter_channel, 3, same_padding=True, bn=bn),
        #                                 Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn),
        #                                 nn.Conv2d(inter_channel, inter_channel, 3, (2, 1), 1))

        self.rec_conv3_1 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_conv3_2 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_pool3_1 = nn.Conv2d(inter_channel, inter_channel, 3, (2, 1), 1)

        # self.max_pool3_1 = nn.MaxPool2d((2, 1), stride=(2, 1))

        # input with shape of [w, b, c] --> [20 timestamps, x fg_nums, 256 channels]
        self.blstm = nn.LSTM(inter_channel, int(inter_channel), bidirectional=True)
        self.embeddings = FC(inter_channel * 2, char_class, relu=None)


    def forward(self, rec_pooled_features):

        rec_x = self.rec_conv1_1(rec_pooled_features)
        rec_x = self.rec_conv1_2(rec_x)
        rec_x = self.rec_pool1_1(rec_x)

        rec_x = self.rec_conv2_1(rec_x)
        rec_x = self.rec_conv2_2(rec_x)
        rec_x = self.rec_pool2_1(rec_x)

        rec_x = self.rec_conv3_1(rec_x)
        rec_x = self.rec_conv3_2(rec_x)
        rec_x = self.rec_pool3_1(rec_x)

        c_feat = rec_x.squeeze(2)
        c_feat = c_feat.permute(2, 0, 1)#.contiguous()

        recurrent, _ = self.blstm(c_feat)
        T, b, h = recurrent.size()
        rec_x = recurrent.view(T * b, h)
        predict = self.embeddings(rec_x)
        predict = predict.view(T, b, -1)

        return predict


class RECG_REFINED(nn.Module):
    def __init__(self, char_class, g_feat_channel=1024, inter_channel=256, bn=True, relu_type="ReLU"):
        super(RECG_REFINED, self).__init__()

        #self.rec_conv1 = nn.Sequential(Conv2dGroup(g_feat_channel, inter_channel, 3, same_padding=True, bn=bn),
        #                              Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn),
        #                              nn.Conv2d(inter_channel, inter_channel, 3, (2, 1), 1))

        # inter_channel *= 2

        activation = Mish() if relu_type == "Mish" else nn.ReLU(inplace=True)

        self.rec_conv1_1 = Conv2dGroup(g_feat_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_conv1_2 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_pool1_1 = nn.Conv2d(inter_channel, inter_channel * 2, 3, (2, 1), 1)

        inter_channel *= 2

        self.rec_conv2_1 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_conv2_2 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_pool2_1 = nn.Conv2d(inter_channel, inter_channel * 2, 3, (2, 1), 1)

        # self.max_pool2_1 = nn.MaxPool2d((2, 1), stride=(2, 1))

        inter_channel *= 2

        # self.rec_conv3 = nn.Sequential(Conv2dGroup(inter_channel // 2, inter_channel, 3, same_padding=True, bn=bn),
        #                                 Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn),
        #                                 nn.Conv2d(inter_channel, inter_channel, 3, (2, 1), 1))

        self.rec_conv3_1 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_conv3_2 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_pool3_1 = nn.Conv2d(inter_channel, inter_channel, 3, (2, 1), 1)

        self.blstm = nn.LSTM(inter_channel, int(inter_channel), bidirectional=True)
        self.embeddings = FC(inter_channel * 2, char_class, relu=None)


    def forward(self, rec_pooled_features):

        rec_x = self.rec_conv1_1(rec_pooled_features)
        rec_x = self.rec_conv1_2(rec_x)
        rec_x = self.rec_pool1_1(rec_x)

        rec_x = self.rec_conv2_1(rec_x)
        rec_x = self.rec_conv2_2(rec_x)
        rec_x = self.rec_pool2_1(rec_x)

        rec_x = self.rec_conv3_1(rec_x)
        rec_x = self.rec_conv3_2(rec_x)

        rec_x = self.rec_pool3_1(rec_x)

        c_feat = rec_x.squeeze(2)
        c_feat = c_feat.permute(2, 0, 1)#.contiguous()

        recurrent, _ = self.blstm(c_feat)
        T, b, h = recurrent.size()

        rec_x = recurrent.view(T * b, h)
        predict = self.embeddings(rec_x)
        predict = predict.view(T, b, -1)

        return predict


class RECG_REFINED_WITH_TRANSFORMER(nn.Module):
    def __init__(self, char_class, g_feat_channel=1024, inter_channel=256, bn=True, relu_type="ReLU"):
        super(RECG_REFINED_WITH_TRANSFORMER, self).__init__()

        activation = Mish() if relu_type == "Mish" else nn.ReLU(inplace=True)

        self.rec_conv1_1 = Conv2dGroup(g_feat_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_conv1_2 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_pool1_1 = nn.Conv2d(inter_channel, inter_channel * 2, 3, (2, 1), 1)

        inter_channel *= 2

        self.rec_conv2_1 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_conv2_2 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_pool2_1 = nn.Conv2d(inter_channel, inter_channel * 2, 3, (1, 1), 1)

        inter_channel *= 2

        # self.rec_conv3_1 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        # self.rec_conv3_2 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        # self.rec_pool3_1 = nn.Conv2d(inter_channel, inter_channel, 3, (2, 1), 1)

        # self.blstm = nn.LSTM(inter_channel, int(inter_channel), bidirectional=True)
        # self.embeddings = FC(inter_channel * 2, char_class, relu=None)


    def forward(self, rec_pooled_features):

        rec_x = self.rec_conv1_1(rec_pooled_features)
        rec_x = self.rec_conv1_2(rec_x)
        rec_x = self.rec_pool1_1(rec_x)

        rec_x = self.rec_conv2_1(rec_x)
        rec_x = self.rec_conv2_2(rec_x)
        rec_x = self.rec_pool2_1(rec_x)

        # rec_x = self.rec_conv3_1(rec_x)
        # rec_x = self.rec_conv3_2(rec_x)
        # rec_x = self.rec_pool3_1(rec_x)

        b = rec_x.size(0)
        c = rec_x.size(1)
        # (n, c, h, w) -> (n, c, h*w) -> (n, t, c)
        # rec_x = rec_x.view(b, c, -1).permute(0, 2, 1)

        return rec_x


class RECG_REFINED_WITHSHORTCURT(nn.Module):
    def __init__(self, char_class, g_feat_channel=1024, inter_channel=256, bn=True, relu_type="ReLU"):
        super(RECG_REFINED_WITHSHORTCURT, self).__init__()

        #self.rec_conv1 = nn.Sequential(Conv2dGroup(g_feat_channel, inter_channel, 3, same_padding=True, bn=bn),
        #                              Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn),
        #                              nn.Conv2d(inter_channel, inter_channel, 3, (2, 1), 1))

        # inter_channel *= 2

        self.activation = activation = Mish() if relu_type == "Mish" else nn.ReLU(inplace=True)

        self.rec_conv1_1 = Conv2dGroup(g_feat_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_conv1_2 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_pool1_1 = nn.Conv2d(inter_channel, inter_channel * 2, 3, (2, 1), 1)

        inter_channel *= 2

        self.rec_conv2_1 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_conv2_2 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_pool2_1 = nn.Conv2d(inter_channel, inter_channel * 2, 3, (2, 1), 1)

        # self.max_pool2_1 = nn.MaxPool2d((2, 1), stride=(2, 1))

        inter_channel *= 2

        # self.rec_conv3 = nn.Sequential(Conv2dGroup(inter_channel // 2, inter_channel, 3, same_padding=True, bn=bn),
        #                                 Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn),
        #                                 nn.Conv2d(inter_channel, inter_channel, 3, (2, 1), 1))

        self.rec_conv3_1 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_conv3_2 = Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, bn=bn, relu=activation)
        self.rec_pool3_1 = nn.Conv2d(inter_channel, inter_channel, 3, (2, 1), 1)

        self.shortcut_pool3_1 = nn.Conv2d(inter_channel, inter_channel * 2, 3, (2, 1), 1)

        # inter_channel *= 2

        # self.max_pool3_1 = nn.MaxPool2d((2, 1), stride=(2, 1))

        # input with shape of [w, b, c] --> [20 timestamps, x fg_nums, 256 channels]

        self.shortcut = nn.LSTM(inter_channel, int(inter_channel * 2), bidirectional=True)

        self.blstm = nn.LSTM(inter_channel, int(inter_channel), bidirectional=True)
        self.embeddings = FC(inter_channel * 2, char_class, relu=None)


    def forward(self, rec_pooled_features):

        rec_x = self.rec_conv1_1(rec_pooled_features)
        rec_x = self.rec_conv1_2(rec_x)
        rec_x = self.rec_pool1_1(rec_x)

        rec_x = self.rec_conv2_1(rec_x)
        rec_x = self.rec_conv2_2(rec_x)
        rec_x = self.rec_pool2_1(rec_x)

        rec_x = self.rec_conv3_1(rec_x)
        rec_x = self.rec_conv3_2(rec_x)

        shortcut_rec_x = self.shortcut_pool3_1(rec_x)
        rec_x = self.rec_pool3_1(rec_x)

        c_feat = rec_x.squeeze(2)
        c_feat = c_feat.permute(2, 0, 1)#.contiguous()

        recurrent, _ = self.blstm(c_feat)
        T, b, h = recurrent.size()

        rec_x = recurrent.view(T * b, h)

        shortcut_rec_x = shortcut_rec_x\
            .squeeze(2)\
            .permute(2, 0, 1)\
            .contiguous()\
            .view(T * b, h)

        add_rec_x = self.activation(shortcut_rec_x + rec_x)

        predict = self.embeddings(add_rec_x)

        predict = predict.view(T, b, -1)

        return predict

RECHEAD_TYPE = {
    "ORIGINAL": RECG,
    "REFINED": RECG_REFINED,
    "REF_SHORTCUT": RECG_REFINED_WITHSHORTCURT,
    "REF_TRANSFORMER": RECG_REFINED_WITH_TRANSFORMER
}

class RRPNRecC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(RRPNRecC4Predictor, self).__init__()

        al_profile = cfg.MODEL.ROI_REC_HEAD.ALPHABET

        if os.path.isfile(al_profile):
            num_classes = len(open(al_profile, 'r').read()) + 1
        else:
            print("We don't expect you to use default class number...Retry it")
            num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.BACKBONE.OUT_CHANNELS

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 3
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        # (c2, c3, c4)
        if cfg.MODEL.FP4P_ON:
            num_inputs = 1024 + 512 + 256

        # input feature size with [N, 1024, 8, 35]
        self.rec_head = RECHEAD_TYPE[cfg.MODEL.ROI_REC_HEAD.STRUCT](num_classes, num_inputs, dim_reduced, cfg.MODEL.ROI_REC_HEAD.ACTIVATION)

        for name, param in self.named_parameters():
            # print('name:', name)
            if "bias" in name:
                nn.init.constant_(param, 0)

            elif "weight" in name and 'bn' in name:
                param.data.fill_(1)
            elif "bias" in name and 'bn' in name:
                param.data.fill_(0)

            elif "weight" in name and not 'gn' in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.rec_head(x)


class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)


_ROI_REC_PREDICTOR = {"MaskRCNNC4Predictor": MaskRCNNC4Predictor,
                       "RRPNE2EC4Predictor": RRPNRecC4Predictor}


def make_roi_rec_predictor(cfg):
    func = _ROI_REC_PREDICTOR[cfg.MODEL.ROI_REC_HEAD.PREDICTOR]
    return func(cfg)
