from os import listdir
from scipy import io
import json
import numpy as np
from polygon_wrapper import iou
from polygon_wrapper import iod



def input_reading(polygons):
    det = []
    for polygon in polygons:
        polygon['points'] = np.array(polygon['points'])
        det.append(polygon)
    return det


def gt_reading(gt_dict, img_key):
    polygons = gt_dict[img_key]
    gt = []
    for polygon in polygons:
        polygon['points'] = np.array(polygon['points'])
        gt.append(polygon)
    return gt


def detection_filtering(detections, groundtruths, threshold=0.5):
    """ignore detected illegal text region"""
    before_filter_num = len(detections)
    for gt_id, gt in enumerate(groundtruths):
        if (gt['transcription'] == '###') and (gt['points'].shape[1] > 1):
            gt_x = list(map(int, np.squeeze(gt['points'][:, 0])))
            gt_y = list(map(int, np.squeeze(gt['points'][:, 1])))
            for det_id, detection in enumerate(detections):
                det_x = list(map(int, np.squeeze(detection['points'][:, 0])))
                det_y = list(map(int, np.squeeze(detection['points'][:, 1])))
                det_gt_iou = iod(det_x, det_y, gt_x, gt_y)
                if det_gt_iou > threshold:
                    detections[det_id] = []

            detections[:] = [item for item in detections if item != []]

    # if before_filter_num - len(detections) > 0:
    #     print("Ignore {} illegal detections".format(before_filter_num - len(detections)))

    return detections


def gt_filtering(groundtruths):
    before_filter_num = len(groundtruths)
    for gt_id, gt in enumerate(groundtruths):
        if gt['transcription'] == '###' or gt['points'].shape[0] < 3:
            groundtruths[gt_id] = []

    groundtruths[:] = [item for item in groundtruths if item != []]

    # if before_filter_num - len(groundtruths) > 0:
    #     print("Ignore {} illegal groundtruths".format(before_filter_num - len(groundtruths)))

    return groundtruths


def eval_func(input_json_path, gt_json_path, iou_threshold = 0.5):
    # load json file as dict
    with open(input_json_path, 'r') as f:
        input_dict = json.load(f)

    with open(gt_json_path, 'r') as f:
        gt_dict = json.load(f)

    # Initial config
    global_tp = 0
    global_fp = 0
    global_fn = 0

    for input_img_key, input_cnts in input_dict.items():
        detections = input_reading(input_cnts)
        groundtruths = gt_reading(gt_dict, input_img_key)  # .replace('res', 'gt'))
        detections = detection_filtering(detections, groundtruths)  # filters detections overlapping with DC area
        groundtruths = gt_filtering(groundtruths)

        iou_table = np.zeros((len(groundtruths), len(detections)))
        det_flag = np.zeros((len(detections), 1))
        gt_flag = np.zeros((len(groundtruths), 1))
        tp = 0
        fp = 0
        fn = 0
        for gt_id, gt in enumerate(groundtruths):
            gt_x = list(map(int, np.squeeze(gt['points'][:, 0])))
            gt_y = list(map(int, np.squeeze(gt['points'][:, 1])))
            if len(detections) > 0:
                for det_id, detection in enumerate(detections):
                    # print(detection['points'])
                    det_x = list(map(int, np.squeeze(detection['points'][:, 0])))
                    det_y = list(map(int, np.squeeze(detection['points'][:, 1])))

                    iou_table[gt_id, det_id] = iou(det_x, det_y, gt_x, gt_y)

                best_matched_det_id = np.argmax(
                    iou_table[gt_id, :])  # identified the best matched detection candidates with current groundtruth

                matched_id = np.where(iou_table[gt_id, :] >= iou_threshold)
                if iou_table[gt_id, best_matched_det_id] >= iou_threshold:
                    if matched_id[0].shape[0] < 2:
                        tp = tp + 1.0
                        global_tp = global_tp + 1.0
                        det_flag[best_matched_det_id] = 1
                        gt_flag[gt_id] = 1
                    else:
                        tp = tp + 1.0
                        global_tp = global_tp + 1.0
                        det_flag[best_matched_det_id] = 1
                        gt_flag[gt_id] = 1
                        # if there are more than 1 matched detection, only 1 is contributed to tp, the rest are fp
                        fp = fp + (matched_id[0].shape[0] - 1.0)

        # Update local and global tp, fp, and fn
        inv_gt_flag = 1 - gt_flag
        fn = np.sum(inv_gt_flag)
        inv_det_flag = 1 - det_flag
        fp = fp + np.sum(inv_det_flag)

        global_fp = global_fp + fp
        global_fn = global_fn + fn
        if tp + fp == 0:
            local_precision = 0
        else:
            local_precision = tp / (tp + fp)

        if tp + fn == 0:
            local_recall = 0
        else:
            local_recall = tp / (tp + fn)

        print('{0:12} Precision: {1:.4f}, Recall: {2:.4f}'.format(input_img_key + '.jpg',  # .replace('res', 'gt')
                                                                  local_precision, local_recall))

    global_precision = global_tp / (global_tp + global_fp)
    global_recall = global_tp / (global_tp + global_fn)
    f_score = 2 * global_precision * global_recall / (global_precision + global_recall)

    print('Global Precision: {:.4f}, Recall: {:.4f}, F_score: {:.4f}'.format(global_precision, global_recall, f_score))

    print('over')


if __name__ == '__main__':
    # Initial config
    global_tp = 0
    global_fp = 0
    global_fn = 0

    input_json_path = 'results/e2e_rrpn_R_50_C4_1x_LSVT_val/model_0190000/res.json'
    gt_json_path = '../datasets/LSVT/train_full_labels.json'
    iou_threshold = 0.3

    # load json file as dict
    with open(input_json_path, 'r') as f:
        input_dict = json.load(f)

    with open(gt_json_path, 'r') as f:
        gt_dict = json.load(f)

    for input_img_key, input_cnts in input_dict.items():
        detections = input_reading(input_cnts)
        groundtruths = gt_reading(gt_dict, input_img_key) #.replace('res', 'gt'))
        detections = detection_filtering(detections, groundtruths)  # filters detections overlapping with DC area
        groundtruths = gt_filtering(groundtruths)

        iou_table = np.zeros((len(groundtruths), len(detections)))
        det_flag = np.zeros((len(detections), 1))
        gt_flag = np.zeros((len(groundtruths), 1))
        tp = 0
        fp = 0
        fn = 0
        for gt_id, gt in enumerate(groundtruths):
            gt_x = list(map(int, np.squeeze(gt['points'][:, 0])))
            gt_y = list(map(int, np.squeeze(gt['points'][:, 1])))
            if len(detections) > 0:
                for det_id, detection in enumerate(detections):
                    # print(detection['points'])
                    det_x = list(map(int, np.squeeze(detection['points'][:, 0])))
                    det_y = list(map(int, np.squeeze(detection['points'][:, 1])))

                    iou_table[gt_id, det_id] = iou(det_x, det_y, gt_x, gt_y)

                best_matched_det_id = np.argmax(
                    iou_table[gt_id, :])  # identified the best matched detection candidates with current groundtruth

                matched_id = np.where(iou_table[gt_id, :] >= iou_threshold)
                if iou_table[gt_id, best_matched_det_id] >= iou_threshold:
                    if matched_id[0].shape[0] < 2:
                        tp = tp + 1.0
                        global_tp = global_tp + 1.0
                        det_flag[best_matched_det_id] = 1
                        gt_flag[gt_id] = 1
                    else:
                        tp = tp + 1.0
                        global_tp = global_tp + 1.0
                        det_flag[best_matched_det_id] = 1
                        gt_flag[gt_id] = 1
                        # if there are more than 1 matched detection, only 1 is contributed to tp, the rest are fp
                        fp = fp + (matched_id[0].shape[0] - 1.0)

        # Update local and global tp, fp, and fn
        inv_gt_flag = 1 - gt_flag
        fn = np.sum(inv_gt_flag)
        inv_det_flag = 1 - det_flag
        fp = fp + np.sum(inv_det_flag)

        global_fp = global_fp + fp
        global_fn = global_fn + fn
        if tp + fp == 0:
            local_precision = 0
        else:
            local_precision = tp / (tp + fp)

        if tp + fn == 0:
            local_recall = 0
        else:
            local_recall = tp / (tp + fn)

        print('{0:12} Precision: {1:.4f}, Recall: {2:.4f}'.format(input_img_key + '.jpg', #.replace('res', 'gt')
                                                                  local_precision, local_recall))

    global_precision = global_tp / (global_tp + global_fp)
    global_recall = global_tp / (global_tp + global_fn)
    f_score = 2 * global_precision * global_recall / (global_precision + global_recall)

    print('Global Precision: {:.4f}, Recall: {:.4f}, F_score: {:.4f}'.format(global_precision, global_recall, f_score))

    print('over')























