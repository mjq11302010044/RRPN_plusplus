import os
import pickle
import torch
import torch.utils.data
from PIL import Image
import sys
import numpy as np
import time
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import json
from maskrcnn_benchmark.data.transforms import transforms as T
from maskrcnn_benchmark.structures.bounding_box import RBoxList
from maskrcnn_benchmark.utils.visualize import vis_image
import cv2
import random
from scipy import io as sio

def poly2rbox_single(poly):
    pt1 = (int(poly[0]), int(poly[1]))
    pt2 = (int(poly[2]), int(poly[3]))
    pt3 = (int(poly[4]), int(poly[5]))
    pt4 = (int(poly[6]), int(poly[7]))

    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

    angle = 0

    if edge1 > edge2:

        width = edge1
        height = edge2
        if pt1[0] - pt2[0] != 0:
            angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
        else:
            angle = 90.0
    elif edge2 >= edge1:
        width = edge2
        height = edge1
        # print pt2[0], pt3[0]
        if pt2[0] - pt3[0] != 0:
            angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
        else:
            angle = 90.0
    if angle < -45.0:
        angle = angle + 180

    x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
    y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

    return [x_ctr, y_ctr, width, height, angle]


def get_ICDAR2013(mode, dataset_dir):
    DATASET_DIR = dataset_dir

    img_dir = "/ch2_training_images/"
    gt_dir = "/ch2_training_localization_transcription_gt"

    # gt_list = []
    # img_list = []

    im_infos = []
    image_dir = DATASET_DIR + img_dir
    gt_file_list = os.listdir(image_dir)

    gt_words = []

    if mode == 'train':
        cache_pkl = './data_cache/IC13_training.pkl'

    if os.path.isfile(cache_pkl):
        return pickle.load(open(cache_pkl, 'rb'))

    for image in gt_file_list:

        prefix = image[:-4]
        im_path = os.path.join(image_dir, image)
        gt_path = os.path.join(dataset_dir + gt_dir, 'gt_' + prefix + '.txt')
        print(im_path)
        gt_list = open(gt_path, 'r', encoding='utf-8').readlines()
        im = cv2.imread(im_path)
        if im is None:
            print(im_path + '--> None')
            continue

        boxes = []
        for gt_ele in gt_list:
            gt_ele = gt_ele.replace('\n', '').replace('\ufeff', '')
            gt = gt_ele.split(',')

            if len(gt) > 1:
                gt_ind = np.array(gt[:8], dtype=np.float32)
                gt_ind = np.array(gt_ind, dtype=np.int32)
                words = gt[8]

                pt1 = (int(gt_ind[0]), int(gt_ind[1]))
                pt2 = (int(gt_ind[2]), int(gt_ind[3]))
                pt3 = (int(gt_ind[4]), int(gt_ind[5]))
                pt4 = (int(gt_ind[6]), int(gt_ind[7]))

                edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

                angle = 0

                if edge1 > edge2:

                    width = edge1
                    height = edge2
                    if pt1[0] - pt2[0] != 0:
                        angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                    else:
                        angle = 90.0
                elif edge2 >= edge1:
                    width = edge2
                    height = edge1
                    # print pt2[0], pt3[0]
                    if pt2[0] - pt3[0] != 0:
                        angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                    else:
                        angle = 90.0
                if angle < -45.0:
                    angle = angle + 180

                x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

                if height * width * (800 / float(im.shape[0])) < 16 * 16 and mode == "train":
                    continue
                # return to width, height
                # if '###' in words:
                #    continue
                boxes.append([x_ctr, y_ctr, width, height, angle, words])
                gt_words.append(words)
        cls_num = 2
        len_of_bboxes = len(boxes)
        gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
        gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = np.zeros((len_of_bboxes, cls_num), dtype=np.float32)  # text or non-text
        seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

        for idx in range(len(boxes)):
            gt_classes[idx] = 1  # cls_text
            overlaps[idx, 1] = 1.0  # prob
            seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])
            gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]

        # print ("boxes_size:", gt_boxes.shape[0])
        if gt_boxes.shape[0] > 0:
            max_overlaps = overlaps.max(axis=1)
            # gt class that had the max overlap
            max_classes = overlaps.argmax(axis=1)
        else:
            continue

        if gt_boxes.shape[0] > 50:
            continue

        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': im_path,
            'boxes': gt_boxes,
            'gt_words': gt_words,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': im.shape[0],
            'width': im.shape[1],
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    f_save_pkl = open(cache_pkl, 'wb')
    pickle.dump(im_infos, f_save_pkl)
    f_save_pkl.close()
    print("Save pickle done.")

    return im_infos


def get_ICDAR2015_RRC_PICK_TRAIN(mode, dataset_dir):
    # dir_path = "/home/shiki-alice/Downloads/ICDAR2015/ch4_training_images/"
    img_file_type = "jpg"

    image_dir = os.path.join(dataset_dir, 'ch4_training_images/')
    gt_dir = os.path.join(dataset_dir, 'ch4_training_localization_transcription_gt/')

    image_list = os.listdir(image_dir)
    image_list.sort()
    im_infos = []

    cache_file = './data_cache/IC15_training.pkl'
    if os.path.isfile(cache_file):
        return pickle.load(open(cache_file, 'rb'))

    for image in image_list:

        prefix = image[:-4]
        img_name = os.path.join(image_dir, image)
        gt_name = os.path.join(gt_dir, 'gt_' + prefix + '.txt')

        # img_name = dir_path + img_list[idx]
        # gt_name = gt_dir + gt_list[idx]

        easy_boxes = []
        hard_boxes = []

        boxes = []
        # print gt_name
        gt_obj = open(gt_name, 'r')
        gt_txt = gt_obj.read()
        gt_split = gt_txt.split('\n')
        img = cv2.imread(img_name)
        print(img_name)
        f = False
        # print '-------------'
        for gt_line in gt_split:

            if not f:
                gt_ind = gt_line.split('\\')

                f = True
            else:
                gt_ind = gt_line.split(',')
            if len(gt_ind) > 3 and '###' not in gt_ind[8]:
                # condinate_list = gt_ind[2].split(',')
                # print ("easy: ", gt_ind)

                pt1 = (int(gt_ind[0]), int(gt_ind[1]))
                pt2 = (int(gt_ind[2]), int(gt_ind[3]))
                pt3 = (int(gt_ind[4]), int(gt_ind[5]))
                pt4 = (int(gt_ind[6]), int(gt_ind[7]))

                edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

                angle = 0

                if edge1 > edge2:

                    width = edge1
                    height = edge2
                    if pt1[0] - pt2[0] != 0:
                        angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                    else:
                        angle = 90.0
                elif edge2 >= edge1:
                    width = edge2
                    height = edge1
                    # print pt2[0], pt3[0]
                    if pt2[0] - pt3[0] != 0:
                        angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                    else:
                        angle = 90.0
                if angle < -45.0:
                    angle = angle + 180

                x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

                easy_boxes.append([x_ctr, y_ctr, width, height, angle])

            if len(gt_ind) > 3 and '###' in gt_ind[8]:
                # condinate_list = gt_ind[2].split(',')

                # print "hard: ", gt_ind

                pt1 = (int(gt_ind[0]), int(gt_ind[1]))
                pt2 = (int(gt_ind[2]), int(gt_ind[3]))
                pt3 = (int(gt_ind[4]), int(gt_ind[5]))
                pt4 = (int(gt_ind[6]), int(gt_ind[7]))

                edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

                angle = 0

                if edge1 > edge2:

                    width = edge1
                    height = edge2
                    if pt1[0] - pt2[0] != 0:
                        angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                    else:
                        angle = 90.0
                elif edge2 >= edge1:
                    width = edge2
                    height = edge1
                    # print pt2[0], pt3[0]
                    if pt2[0] - pt3[0] != 0:
                        angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                    else:
                        angle = 90.0
                if angle < -45.0:
                    angle = angle + 180

                x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

                if angle > 45. and angle < 135:
                    continue

                # if (width / float(height) < 2. or width / float(height) > 8.) and :
                #     continue

                hard_boxes.append([x_ctr, y_ctr, width, height, angle])

        boxes.extend(easy_boxes)

        all_len = len(easy_boxes) + int(len(hard_boxes))
        if all_len <= 50:
            boxes.extend(hard_boxes[0: int(len(hard_boxes) / 5)])
        else:
            print("boxes 50:", all_len, len(easy_boxes), len(hard_boxes))
        # boxes.extend(hard_boxes[0: int(len(hard_boxes))])

        len_of_bboxes = len(boxes)
        gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
        gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32)  # text or non-text
        seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

        for idx in range(len(boxes)):
            gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
            gt_classes[idx] = 1  # cls_text
            overlaps[idx, 1] = 1.0  # cls_text
            seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])

        max_overlaps = overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = overlaps.argmax(axis=1)
        if gt_boxes.shape[0] <= 0: # or gt_boxes.shape[0] >= 50
            continue
        # print('gt_boxes:', gt_boxes)
        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': img_name,
            'boxes': gt_boxes,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': img.shape[0],
            'width': img.shape[1],
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    f_save_pkl = open(cache_file, 'wb')
    pickle.dump(im_infos, f_save_pkl)
    f_save_pkl.close()
    print ("Save pickle done.")

    return im_infos


def get_ICDAR2017_mlt(mode, dataset_dir):
    DATASET_DIR = dataset_dir
    task = 'double_class'
    prefetched = True if os.path.isfile('./data_cache/ICDAR2017_' + mode + '_cache.pkl') else False
    im_infos = []

    data_list = []
    gt_list = []
    img_type = ['jpg', 'png', 'gif']
    cls_list = {'background': 0, 'Arabic': 1, 'English': 2, 'Japanese': 3, 'French': 4, 'German': 5, 'Chinese': 6,
                'Korean': 7, 'Italian': 8, 'Bangla': 9}

    if not prefetched:
        # training set contains 7200 images with
        if mode == "train":
            for i in range(7200):
                img_candidate_path = DATASET_DIR + "ch8_training_images_" + str(int(i / 1000) + 1) + "/" + 'img_' + str(
                    i + 1) + "."
                if os.path.isfile(img_candidate_path + img_type[0]):
                    img_candidate_path += img_type[0]
                elif os.path.isfile(img_candidate_path + img_type[1]):
                    img_candidate_path += img_type[1]
                elif os.path.isfile(img_candidate_path + img_type[2]):
                    im = Image.open(img_candidate_path + img_type[2])
                    im = im.convert('RGB')
                    im.save(img_candidate_path + "jpg", "jpeg")
                    img_candidate_path = img_candidate_path + "jpg"
                data_list.append(img_candidate_path)
                # print ("data_list:", len(data_list))

                gt_candidate_path = DATASET_DIR + "ch8_training_localization_transcription_gt_v2/" + 'gt_img_' + str(
                    i + 1) + ".txt"
                if os.path.isfile(gt_candidate_path):
                    gt_list.append(gt_candidate_path)
                # print ("gt_list:", len(gt_list))

                f_gt = open(gt_candidate_path)
                f_content = f_gt.read()

                lines = f_content.split('\n')
                print (img_candidate_path)
                img = cv2.imread(img_candidate_path)
                boxes = []

                easy_boxes = []
                hard_boxes = []

                for gt_line in lines:
                    # print (gt_line)
                    gt_ind = gt_line.split(',')

                    if len(gt_ind) > 3:

                        pt1 = (int(gt_ind[0]), int(gt_ind[1]))
                        pt2 = (int(gt_ind[2]), int(gt_ind[3]))
                        pt3 = (int(gt_ind[4]), int(gt_ind[5]))
                        pt4 = (int(gt_ind[6]), int(gt_ind[7]))

                        edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                        edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

                        angle = 0

                        if edge1 > edge2:

                            width = edge1
                            height = edge2
                            if pt1[0] - pt2[0] != 0:
                                angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                            else:
                                angle = 90.0
                        elif edge2 >= edge1:
                            width = edge2
                            height = edge1
                            # print pt2[0], pt3[0]
                            if pt2[0] - pt3[0] != 0:
                                angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                            else:
                                angle = 90.0
                        if angle < -45.0:
                            angle = angle + 180

                        x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                        y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

                        if height * width < 8 * 8:
                             continue

                        if not gt_ind[8].replace('\n', '') in ['English', 'French', 'German', 'Italian']:
                            continue

                        if "###" in gt_ind[9]:
                            hard_boxes.append([x_ctr, y_ctr, width, height, angle, gt_ind[8]])
                        else:
                            easy_boxes.append([x_ctr, y_ctr, width, height, angle, gt_ind[8]])

                boxes.extend(easy_boxes)

                all_len = len(easy_boxes) + int(len(hard_boxes))
                # if all_len <= 50:
                boxes.extend(hard_boxes[0: int(len(hard_boxes) * 0.2)])
                # else:
                #     print("boxes 50:", all_len, len(easy_boxes), len(hard_boxes))
                #    boxes.extend(hard_boxes[0: int(len(hard_boxes))])

                # boxes.extend(easy_boxes)
                # boxes.extend(hard_boxes[0: int(len(hard_boxes) * 0.8)])

                # print ("line_size:", len(lines))

                cls_num = 2
                if task == "multi_class":
                    cls_num = len(cls_list.keys())

                len_of_bboxes = len(boxes)
                gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
                gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
                overlaps = np.zeros((len_of_bboxes, cls_num), dtype=np.float32)  # text or non-text
                seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

                if task == "multi_class":
                    gt_boxes = []  # np.zeros((len_of_bboxes, 5), dtype=np.int16)
                    gt_classes = []  # np.zeros((len_of_bboxes), dtype=np.int32)
                    overlaps = []  # np.zeros((len_of_bboxes, cls_num), dtype=np.float32) #text or non-text
                    seg_areas = []  # np.zeros((len_of_bboxes), dtype=np.float32)

                for idx in range(len(boxes)):

                    if task == "multi_class":
                        if not boxes[idx][5] in cls_list:
                            print (boxes[idx][5] + " not in list")
                            continue
                        gt_classes.append(cls_list[boxes[idx][5]])  # cls_text
                        overlap = np.zeros((cls_num))
                        overlap[cls_list[boxes[idx][5]]] = 1.0  # prob
                        overlaps.append(overlap)
                        seg_areas.append((boxes[idx][2]) * (boxes[idx][3]))
                        gt_boxes.append([boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]])
                    else:
                        gt_classes[idx] = 1  # cls_text
                        overlaps[idx, 1] = 1.0  # prob
                        seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])
                        gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]

                if task == "multi_class":
                    gt_classes = np.array(gt_classes)
                    overlaps = np.array(overlaps)
                    seg_areas = np.array(seg_areas)
                    gt_boxes = np.array(gt_boxes)

                # print ("boxes_size:", gt_boxes.shape[0])
                if gt_boxes.shape[0] > 0:
                    max_overlaps = overlaps.max(axis=1)
                    # gt class that had the max overlap
                    max_classes = overlaps.argmax(axis=1)
                else:
                    continue

                if gt_boxes.shape[0] <= 0:
                    print("zero shape:", img_candidate_path)
                    continue

                im_info = {
                    'gt_classes': gt_classes,
                    'max_classes': max_classes,
                    'image': img_candidate_path,
                    'boxes': gt_boxes,
                    'flipped': False,
                    'gt_overlaps': overlaps,
                    'seg_areas': seg_areas,
                    'height': img.shape[0],
                    'width': img.shape[1],
                    'max_overlaps': max_overlaps,
                    'rotated': True
                }
                im_infos.append(im_info)

            f_save_pkl = open('./data_cache/ICDAR2017_' + mode + '_cache.pkl', 'wb')
            pickle.dump(im_infos, f_save_pkl)
            f_save_pkl.close()
            print ("Save pickle done.")
        elif mode == "validation":
            for i in range(1800):
                img_candidate_path = DATASET_DIR + "ch8_validation_images/" + 'img_' + str(i + 1) + "."
                if os.path.isfile(img_candidate_path + img_type[0]):
                    img_candidate_path += img_type[0]
                elif os.path.isfile(img_candidate_path + img_type[1]):
                    img_candidate_path += img_type[1]
                elif os.path.isfile(img_candidate_path + img_type[2]):
                    im = Image.open(img_candidate_path + img_type[2])
                    im = im.convert('RGB')
                    im.save(img_candidate_path + "jpg", "jpeg")
                    img_candidate_path = img_candidate_path + "jpg"
                data_list.append(img_candidate_path)
                # print ("data_list:", len(data_list))

                gt_candidate_path = DATASET_DIR + "ch8_validation_localization_transcription_gt_v2/" + 'gt_img_' + str(
                    i + 1) + ".txt"
                if os.path.isfile(gt_candidate_path):
                    gt_list.append(gt_candidate_path)
                # print ("gt_list:", len(gt_list))

                f_gt = open(gt_candidate_path)
                f_content = f_gt.read()

                lines = f_content.split('\n')
                print (img_candidate_path)
                img = cv2.imread(img_candidate_path)
                boxes = []

                hard_boxes = []
                easy_boxes = []

                for gt_line in lines:
                    # print (gt_line)
                    gt_ind = gt_line.split(',')
                    if len(gt_ind) > 3:

                        pt1 = (int(gt_ind[0]), int(gt_ind[1]))
                        pt2 = (int(gt_ind[2]), int(gt_ind[3]))
                        pt3 = (int(gt_ind[4]), int(gt_ind[5]))
                        pt4 = (int(gt_ind[6]), int(gt_ind[7]))

                        edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                        edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

                        angle = 0

                        if edge1 > edge2:

                            width = edge1
                            height = edge2
                            if pt1[0] - pt2[0] != 0:
                                angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                            else:
                                angle = 90.0
                        elif edge2 >= edge1:
                            width = edge2
                            height = edge1
                            # print pt2[0], pt3[0]
                            if pt2[0] - pt3[0] != 0:
                                angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                            else:
                                angle = 90.0
                        if angle < -45.0:
                            angle = angle + 180

                        x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                        y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

                        if height * width * (800 / float(img.shape[0])) < 16 * 16:
                            continue

                        if not gt_ind[8].replace('\n', '') in ['English', 'French', 'German', 'Italian']:
                            continue

                        # boxes.append([x_ctr, y_ctr, width, height, angle, gt_ind[8]])

                        if "###" in gt_ind[9]:
                            hard_boxes.append([x_ctr, y_ctr, width, height, angle, gt_ind[8]])
                        else:
                            easy_boxes.append([x_ctr, y_ctr, width, height, angle, gt_ind[8]])

                boxes.extend(easy_boxes)

                all_len = len(easy_boxes) + int(len(hard_boxes))
                # if all_len <= 50:
                boxes.extend(hard_boxes[0: int(len(hard_boxes) * 0.2)])
                # else:
                #     print("boxes 50:", all_len, len(easy_boxes), len(hard_boxes))
                #     boxes.extend(hard_boxes[0: int(len(hard_boxes))])

                cls_num = 2
                if task == "multi_class":
                    cls_num = len(cls_list.keys())

                len_of_bboxes = len(boxes)
                gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
                gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
                overlaps = np.zeros((len_of_bboxes, cls_num), dtype=np.float32)  # text or non-text
                seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

                for idx in range(len(boxes)):

                    if task == "multi_class":
                        if not boxes[idx][5] in cls_list:
                            break
                        gt_classes[idx] = cls_list[boxes[idx][5]]  # cls_text
                        overlaps[idx, cls_list[boxes[idx][5]]] = 1.0  # prob
                    else:
                        gt_classes[idx] = 1  # cls_text
                        overlaps[idx, 1] = 1.0  # prob
                    seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])
                    gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]

                max_overlaps = overlaps.max(axis=1)
                # gt class that had the max overlap
                max_classes = overlaps.argmax(axis=1)

                if gt_boxes.shape[0] <= 0 or gt_boxes.shape[0] >= 50:
                    continue

                im_info = {
                    'gt_classes': gt_classes,
                    'max_classes': max_classes,
                    'image': img_candidate_path,
                    'boxes': gt_boxes,
                    'flipped': False,
                    'gt_overlaps': overlaps,
                    'seg_areas': seg_areas,
                    'height': img.shape[0],
                    'width': img.shape[1],
                    'max_overlaps': max_overlaps,
                    'rotated': True
                }
                im_infos.append(im_info)

            f_save_pkl = open('./data_cache/ICDAR2017_' + mode + '_cache.pkl', 'wb')
            pickle.dump(im_infos, f_save_pkl)
            f_save_pkl.close()
            print ("Save pickle done.")
    else:
        if mode == "train":
            f_pkl = open('./data_cache/ICDAR2017_' + mode + '_cache.pkl', 'rb')
            im_infos = pickle.load(f_pkl)
        if mode == "validation":
            f_pkl = open('./data_cache/ICDAR2017_' + mode + '_cache.pkl', 'rb')
            im_infos = pickle.load(f_pkl)
    return im_infos


def get_Syn_800K_with_words(mode, dataset_dir):
    # if mode == 'train':
    #    image_dir = os.path.join(dataset_dir, 'image_9000/')
    # gt_dir = os.path.join(dataset_dir, 'txt_9000/')

    # ./ICPR_dataset/update_ICPR_text_train_part1_20180316/train_1000/
    # else:
    #    image_dir = os.path.join(dataset_dir, 'image_1000/')
    # gt_dir = os.path.join(dataset_dir, 'txt_1000/')

    word2vec_mat = '../selected_smaller_dic.mat'
    #mat_data = sio.loadmat(word2vec_mat)
    #all_words = mat_data['selected_vocab']
    #all_vecs = mat_data['selected_dict']

    #w2v_dict = {}
    #print('Building w2v dictionary...')
    #for i in range(len(all_words)):
    #    w2v_dict[all_words[i][0][0]] = all_vecs[i]
    #print('done')

    mat_file = os.path.join(dataset_dir, 'gt.mat')
    # print('mat_file:', mat_file)
    mat_f = sio.loadmat(mat_file)

    wordBBs = mat_f['wordBB'][0]
    txt_annos = mat_f['txt'][0]
    im_names = mat_f['imnames'][0]

    sam_size = len(txt_annos)

    # image_list = os.listdir(image_dir)
    # image_list.sort()
    im_infos = []

    if mode == 'train':
        cache_pkl = './data_cache/Syn_800K_training.pkl'
    else:
        cache_pkl = './data_cache/Syn_800K_testing.pkl'

    if os.path.isfile(cache_pkl):
        return pickle.load(open(cache_pkl, 'rb'))

    pro_cnt = 0

    for i in range(sam_size):
        txts = txt_annos[i]
        im_path = os.path.join(dataset_dir, im_names[i][0])
        word_boxes = wordBBs[i]

        pro_cnt += 1
        if pro_cnt % 200 == 0:
            print('processed image:', str(pro_cnt) + '/' + str(sam_size))

        cnt = 0
        # print('word_boxes:', word_boxes.shape)
        im = cv2.imread(im_path)

        if len(word_boxes.shape) < 3:
            word_boxes = np.expand_dims(word_boxes, -1)
        words = []
        boxes = []
        word_vecs = []

        for txt in txts:
            txtsp = txt.split('\n')
            for line in txtsp:
                line = line.replace('\n', '').replace('\n', '').replace('\r', '').replace('\t', '').split(' ')
                # print('line:', line)
                for w in line:
                    # w = w
                    if len(w) > 0:
                        gt_ind = np.transpose(np.array(word_boxes[:, :, cnt], dtype=np.int32), (1, 0)).reshape(8)
                        # print(imname, gt_ind, w)
                        cnt += 1
                        '''
                        cv2.line(im, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
                        cv2.line(im, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 3)
                        cv2.line(im, (box[4], box[5]), (box[6], box[7]), (0, 0, 255), 3)
                        cv2.line(im, (box[6], box[7]), (box[0], box[1]), (0, 0, 255), 3)
                        cv2.putText(im, w, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 122), 2)
                        '''

                        pt1 = (int(gt_ind[0]), int(gt_ind[1]))
                        pt2 = (int(gt_ind[2]), int(gt_ind[3]))
                        pt3 = (int(gt_ind[4]), int(gt_ind[5]))
                        pt4 = (int(gt_ind[6]), int(gt_ind[7]))

                        edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                        edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

                        angle = 0

                        if edge1 > edge2:

                            width = edge1
                            height = edge2
                            if pt1[0] - pt2[0] != 0:
                                angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                            else:
                                angle = 90.0
                        elif edge2 >= edge1:
                            width = edge2
                            height = edge1
                            # print pt2[0], pt3[0]
                            if pt2[0] - pt3[0] != 0:
                                angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                            else:
                                angle = 90.0
                        if angle < -45.0:
                            angle = angle + 180

                        x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                        y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

                        if height * width < 8 * 8 and mode == "train":
                            continue
                        if x_ctr >= im.shape[1] or x_ctr < 0 or y_ctr >= im.shape[0] or y_ctr < 0:
                            continue

                        #com_num = re.compile('[0-9]+')
                        #com_prices = re.compile('[$￥€£]+')

                        #match_num = re.findall(com_num, w)
                        #match_prices = re.findall(com_prices, w)

                        # choices: original, prices, others
                        words.append(w)
                        '''
                        w = w.lower()
                        if w in w2v_dict:
                            word_vecs.append(w2v_dict[w.lower()])
                        elif match_prices and match_num:
                            word_vecs.append(w2v_dict['price'])
                        elif match_num and not match_prices:
                            word_vecs.append(w2v_dict['ten'])
                        else:
                            print(im_path, w)
                            word_vecs.append(np.zeros(100, dtype=np.float32) + 1e-10)
                        '''
                        # return to width, height
                        boxes.append([x_ctr, y_ctr, width, height, angle, w])
        cls_num = 2
        len_of_bboxes = len(boxes)
        gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
        gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = np.zeros((len_of_bboxes, cls_num), dtype=np.float32)  # text or non-text
        seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

        for idx in range(len(boxes)):
            gt_classes[idx] = 1  # cls_text
            overlaps[idx, 1] = 1.0  # prob
            seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])
            gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]

        # print ("boxes_size:", gt_boxes.shape[0])
        if gt_boxes.shape[0] > 0:
            max_overlaps = overlaps.max(axis=1)
            # gt class that had the max overlap
            max_classes = overlaps.argmax(axis=1)
        else:
            continue

        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': im_path,
            'boxes': gt_boxes,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': im.shape[0],
            'width': im.shape[1],
            'gt_words': words,
            # 'gt_wordvec': np.array(word_vecs),
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    f_save_pkl = open(cache_pkl, 'wb')
    pickle.dump(im_infos, f_save_pkl)
    f_save_pkl.close()
    print("Save pickle done.")
    return im_infos


def get_ICDAR_LSVT_full(mode, dataset_dir):

    assert mode in ['train', 'val', 'full'], 'mode not in ' + str(['train', 'val', 'full'])

    data_split = {
        'val':[0, 3000],
        'train':[3000, 30000],
        'full':[0, 30000]
    }

    vis = False

    cache_file = './data_cache/LSVT_det_' + mode + '.pkl'
    if os.path.isfile(cache_file):
        print('dataset cache found, loading from it...')
        im_infos = pickle.load(open(cache_file, 'rb'))
        print('load done')
        return im_infos

    im_codes = range(data_split[mode][0], data_split[mode][1])
    gt_json = os.path.join(dataset_dir, 'train_full_labels.json')

    gt_dict = json.load(open(gt_json, 'r'))

    im_infos = []

    num_samples = data_split[mode][1] - data_split[mode][0]

    for imnum in im_codes:
        forder = int(imnum / 15000)
        imfolder = os.path.join(dataset_dir, 'train_full_images_'+str(forder))
        impath = os.path.join(imfolder, 'gt_' + str(imnum) + '.jpg')
        gt_code = 'gt_' + str(imnum)
        gt_anno = gt_dict[gt_code]

        inst_num = len(gt_anno)

        im = cv2.imread(impath)

        easy_boxes = []
        hard_boxes = []

        print(str(imnum) + '/' + str(data_split[mode][0] + num_samples), impath)

        for i in range(inst_num):
            inst = gt_anno[i]
            # print(inst.keys())
            poly = np.array(inst['points'])
            words = inst['transcription']
            illegibility = inst['illegibility']

            color = (255, 0, 255) if illegibility else (0, 0, 255)
            if poly.shape[0] > 4:
                # print('polygon:', poly.shape[0])
                rect = cv2.minAreaRect(poly)
                poly = np.array(cv2.boxPoints(rect), np.int)
                # print('rect:', rect)
                if vis:
                    rect_pt_num = rect.shape[0]
                    for i in range(rect.shape[0]):
                        cv2.line(im, (rect[i % rect_pt_num][0], rect[i % rect_pt_num][1]),
                                 (rect[(i + 1) % rect_pt_num][0], rect[(i + 1) % rect_pt_num][1]), (0, 255, 0), 2)
            if vis:
                pt_num = poly.shape[0]
                for i in range(poly.shape[0]):
                    cv2.line(im, (poly[i % pt_num][0], poly[i % pt_num][1]),
                             (poly[(i + 1) % pt_num][0], poly[(i + 1) % pt_num][1]), color, 2)

            poly = poly.reshape(-1)
            pt1 = (int(poly[0]), int(poly[1]))
            pt2 = (int(poly[2]), int(poly[3]))
            pt3 = (int(poly[4]), int(poly[5]))
            pt4 = (int(poly[6]), int(poly[7]))

            edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
            edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

            angle = 0

            if edge1 > edge2:

                width = edge1
                height = edge2
                if pt1[0] - pt2[0] != 0:
                    angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                else:
                    angle = 90.0
            elif edge2 >= edge1:
                width = edge2
                height = edge1
                # print pt2[0], pt3[0]
                if pt2[0] - pt3[0] != 0:
                    angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                else:
                    angle = 90.0
            if angle < -45.0:
                angle = angle + 180

            x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
            y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

            # if height * width * (800 / float(img.shape[0])) < 16 * 16 and mode == "train":
            #    continue

            if illegibility:
                hard_boxes.append([x_ctr, y_ctr, width, height, angle])
            else:
                easy_boxes.append([x_ctr, y_ctr, width, height, angle])

            # boxes.append([x_ctr, y_ctr, width, height, angle, gt_ind[8]])
        # img_pil = Image.fromarray(im)
        boxes = []
        boxes.extend(easy_boxes)
        boxes.extend(hard_boxes[0: int(len(hard_boxes) / 5)])

        len_of_bboxes = len(boxes)
        gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
        gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32)  # text or non-text
        seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

        for idx in range(len(boxes)):
            gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
            gt_classes[idx] = 1  # cls_text
            overlaps[idx, 1] = 1.0  # cls_text
            seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])

        # img_pil = vis_image(img_pil, gt_boxes)
        # img_pil.save('gt_LSVT.jpg', 'jpeg')
        # break
        max_overlaps = overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = overlaps.argmax(axis=1)
        if gt_boxes.shape[0] <= 0 or gt_boxes.shape[0] > 100:
            continue
        # print('gt_boxes:', gt_boxes)
        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': impath,
            'boxes': gt_boxes,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': im.shape[0],
            'width': im.shape[1],
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    print('Saving pkls...')
    pkl_f = open(cache_file, 'wb')
    pickle.dump(im_infos, pkl_f)
    pkl_f.close()
    print('done')
    return im_infos


def get_ICDAR_ReCTs_full(mode, dataset_dir):

    assert mode in ['train', 'val', 'full'], 'mode not in ' + str(['train', 'val', 'full'])

    data_split = {
        'val':[0, 3000],
        'train':[0, 18000],
        'full':[0, 30000]
    }

    vis = False

    cache_file = './data_cache/ReCTs_det_' + mode + '.pkl'
    if os.path.isfile(cache_file):
        print('dataset cache found, loading from it...')
        im_infos = pickle.load(open(cache_file, 'rb'))
        print('load done')
        return im_infos

    # im_codes = range(data_split[mode][0], data_split[mode][1])
    # gt_json = os.path.join(dataset_dir, 'train_full_labels.json')

    # gt_dict = json.load(open(gt_json, 'r'))

    gt_dir = os.path.join(dataset_dir, mode, 'gt')
    im_dir = os.path.join(dataset_dir, mode, 'image')

    imlist = os.listdir(im_dir)

    im_infos = []

    num_samples = data_split[mode][1] - data_split[mode][0]

    cnt = 0

    for imname in imlist:
        # forder = int(imnum / 15000)
        # imfolder = os.path.join(dataset_dir, 'train_full_images_'+str(forder), 'train_full_images_'+str(forder))
        impath = os.path.join(im_dir, imname)
        gtpath = os.path.join(gt_dir, imname.split('.')[0] + '.json')
        gt_anno = open(gtpath, 'r')

        inst_num = len(gt_anno)

        im = cv2.imread(impath)

        easy_boxes = []
        hard_boxes = []
        cnt += 1
        print(str(cnt) + '/' + str(data_split[mode][0] + num_samples), impath)

        # using lines
        lines = gt_anno['lines']

        for i in range(len(lines)):
            inst = lines[i]
            # print(inst.keys())
            poly = np.array(inst['points']).reshape(-1, 2)
            words = inst['transcription']
            ignore = inst['ignore']

            color = (255, 0, 255) if not ignore else (0, 0, 255)
            if poly.shape[0] > 4:
                # print('polygon:', poly.shape[0])
                rect = cv2.minAreaRect(poly)
                poly = np.array(cv2.boxPoints(rect), np.int)
                # print('rect:', rect)
                if vis:
                    rect_pt_num = rect.shape[0]
                    for i in range(rect.shape[0]):
                        cv2.line(im, (rect[i % rect_pt_num][0], rect[i % rect_pt_num][1]),
                                 (rect[(i + 1) % rect_pt_num][0], rect[(i + 1) % rect_pt_num][1]), (0, 255, 0), 2)
            if vis:
                pt_num = poly.shape[0]
                for i in range(poly.shape[0]):
                    cv2.line(im, (poly[i % pt_num][0], poly[i % pt_num][1]),
                             (poly[(i + 1) % pt_num][0], poly[(i + 1) % pt_num][1]), color, 2)

            poly = poly.reshape(-1)
            pt1 = (int(poly[0]), int(poly[1]))
            pt2 = (int(poly[2]), int(poly[3]))
            pt3 = (int(poly[4]), int(poly[5]))
            pt4 = (int(poly[6]), int(poly[7]))

            edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
            edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

            angle = 0

            if edge1 > edge2:

                width = edge1
                height = edge2
                if pt1[0] - pt2[0] != 0:
                    angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                else:
                    angle = 90.0
            elif edge2 >= edge1:
                width = edge2
                height = edge1
                # print pt2[0], pt3[0]
                if pt2[0] - pt3[0] != 0:
                    angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                else:
                    angle = 90.0
            if angle < -45.0:
                angle = angle + 180

            x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
            y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

            # if height * width * (800 / float(img.shape[0])) < 16 * 16 and mode == "train":
            #    continue

            if ignore:
                hard_boxes.append([x_ctr, y_ctr, width, height, angle])
            else:
                easy_boxes.append([x_ctr, y_ctr, width, height, angle])

            # boxes.append([x_ctr, y_ctr, width, height, angle, gt_ind[8]])
        # img_pil = Image.fromarray(im)
        boxes = []
        boxes.extend(easy_boxes)
        boxes.extend(hard_boxes[0: int(len(hard_boxes) / 5)])

        len_of_bboxes = len(boxes)
        gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
        gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32)  # text or non-text
        seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

        for idx in range(len(boxes)):
            gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
            gt_classes[idx] = 1  # cls_text
            overlaps[idx, 1] = 1.0  # cls_text
            seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])

        # img_pil = vis_image(img_pil, gt_boxes)
        # img_pil.save('gt_LSVT.jpg', 'jpeg')
        # break
        max_overlaps = overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = overlaps.argmax(axis=1)
        if gt_boxes.shape[0] <= 0 or gt_boxes.shape[0] > 100:
            continue
        # print('gt_boxes:', gt_boxes)
        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': impath,
            'boxes': gt_boxes,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': im.shape[0],
            'width': im.shape[1],
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    print('Saving pkls...')
    pkl_f = open(cache_file, 'wb')
    pickle.dump(im_infos, pkl_f)
    pkl_f.close()
    print('done')
    return im_infos



def get_ICDAR_ArT(mode, dataset_dir):

    assert mode in ['train', 'val', 'full'], 'mode not in ' + str(['train', 'val', 'full'])

    data_split = {
        'val':[4000, 5603],
        'train':[0, 4000],
        'full':[0, 5603]
    }

    vis = False

    dataset_dir = os.path.join(dataset_dir, 'ArT_detect_train')

    cache_file = './data_cache/ArT_det_' + mode + '.pkl'
    if os.path.isfile(cache_file):
        print('dataset cache found, loading from it...')
        im_infos = pickle.load(open(cache_file, 'rb'))
        print('load done')
        return im_infos

    im_codes = range(data_split[mode][0], data_split[mode][1])
    gt_json = os.path.join(dataset_dir, 'train_labels.json')

    gt_dict = json.load(open(gt_json, 'r'))

    im_infos = []

    num_samples = data_split[mode][1] - data_split[mode][0]

    for imnum in im_codes:
        # forder = int(imnum / 15000)
        imfolder = os.path.join(dataset_dir, 'train_images')
        impath = os.path.join(imfolder, 'gt_' + str(imnum) + '.jpg')
        gt_code = 'gt_' + str(imnum)
        gt_anno = gt_dict[gt_code]

        inst_num = len(gt_anno)

        im = cv2.imread(impath)

        easy_boxes = []
        hard_boxes = []

        print(str(imnum) + '/' + str(data_split[mode][0] + num_samples), impath)

        for i in range(inst_num):
            inst = gt_anno[i]
            # print(inst.keys())
            poly = np.array(inst['points'])
            words = inst['transcription']
            illegibility = inst['illegibility']
            language = inst['language']

            color = (255, 0, 255) if illegibility else (0, 0, 255)
            if poly.shape[0] > 4:
                # print('polygon:', poly.shape[0])
                rect = cv2.minAreaRect(poly)
                poly = np.array(cv2.boxPoints(rect), np.int)
                # print('rect:', rect)
                if vis:
                    rect_pt_num = rect.shape[0]
                    for i in range(rect.shape[0]):
                        cv2.line(im, (rect[i % rect_pt_num][0], rect[i % rect_pt_num][1]),
                                 (rect[(i + 1) % rect_pt_num][0], rect[(i + 1) % rect_pt_num][1]), (0, 255, 0), 2)
            if vis:
                pt_num = poly.shape[0]
                for i in range(poly.shape[0]):
                    cv2.line(im, (poly[i % pt_num][0], poly[i % pt_num][1]),
                             (poly[(i + 1) % pt_num][0], poly[(i + 1) % pt_num][1]), color, 2)

            if poly.shape[0] < 4:
                print('poly:', poly.shape, np.array(inst['points']).shape)
                continue
            poly = poly.reshape(-1)
            pt1 = (int(poly[0]), int(poly[1]))
            pt2 = (int(poly[2]), int(poly[3]))
            pt3 = (int(poly[4]), int(poly[5]))
            pt4 = (int(poly[6]), int(poly[7]))

            edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
            edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

            angle = 0

            if edge1 > edge2:

                width = edge1
                height = edge2
                if pt1[0] - pt2[0] != 0:
                    angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                else:
                    angle = 90.0
            elif edge2 >= edge1:
                width = edge2
                height = edge1
                # print pt2[0], pt3[0]
                if pt2[0] - pt3[0] != 0:
                    angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                else:
                    angle = 90.0
            if angle < -45.0:
                angle = angle + 180

            x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
            y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

            # if height * width * (800 / float(img.shape[0])) < 16 * 16 and mode == "train":
            #     continue

            if illegibility:
                hard_boxes.append([x_ctr, y_ctr, width, height, angle])
            else:
                easy_boxes.append([x_ctr, y_ctr, width, height, angle])

            # boxes.append([x_ctr, y_ctr, width, height, angle, gt_ind[8]])
        # img_pil = Image.fromarray(im)
        boxes = []
        boxes.extend(easy_boxes)
        # boxes.extend(hard_boxes[0: int(len(hard_boxes) / 3)])

        len_of_bboxes = len(boxes)
        gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
        gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32)  # text or non-text
        seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

        for idx in range(len(boxes)):
            gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
            gt_classes[idx] = 1  # cls_text
            overlaps[idx, 1] = 1.0  # cls_text
            seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])

        # img_pil = vis_image(img_pil, gt_boxes)
        # img_pil.save('gt_LSVT.jpg', 'jpeg')
        # break
        max_overlaps = overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = overlaps.argmax(axis=1)
        if gt_boxes.shape[0] <= 0:
            continue
        # print('gt_boxes:', gt_boxes)
        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': impath,
            'boxes': gt_boxes,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': im.shape[0],
            'width': im.shape[1],
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    print('Saving pkls...')
    pkl_f = open(cache_file, 'wb')
    pickle.dump(im_infos, pkl_f)
    pkl_f.close()
    print('done')
    return im_infos


def get_docVQA(mode, dataset_dir):

    cache_file = './data_cache/docVQA_det_' + mode + '.pkl'
    if os.path.isfile(cache_file):
        print('dataset cache found, loading from it...')
        im_infos = pickle.load(open(cache_file, 'rb'))
        print('load done')
        return im_infos

    gt_dir = os.path.join(dataset_dir, "./train_v0.1_documents_ocr")
    im_dir = os.path.join(dataset_dir, "./docvqa_train_v0.1/documents")

    gt_flist = os.listdir(gt_dir)

    gt_flist.sort()

    picked_gt_flist = []
    im_infos = []
    # Total in 4331
    if mode == "train":
        picked_gt_flist = gt_flist[:4000]
    elif mode == "validation":
        picked_gt_flist = gt_flist[4000:]
    elif mode == "whole":
        picked_gt_flist = gt_flist

    flen = len(picked_gt_flist)

    cnt = 0

    for gt_json in picked_gt_flist:

        gt_path = os.path.join(gt_dir, gt_json)

        imprefix = gt_json.split(".")[0]
        impath = os.path.join(im_dir, imprefix + ".png")
        # im = cv2.imread(impath)
        cnt += 1
        print(impath, str(cnt) + "/" + str(flen))

        json_dict = json.load(open(gt_path, "r"))

        # print("json_dict:", json_dict.keys())

        boxes = []
        easy_boxes = []
        hard_boxes = []

        for recRes in json_dict["recognitionResults"]:

            # print("json_dict:", recRes)
            # print()

            for line in recRes["lines"]:

                # print("line:", line)

                line_box = np.array(line["boundingBox"])
                line_text = line["text"]
                # print("linekey:", line.keys())
                # im = draw_boxes(im, line_box, (255, 0, 0))

                for word in line["words"]:
                    # print("wordkey:", word.keys())
                    word_box = np.array(word["boundingBox"])
                    word_text = word["text"]

                    poly = word_box

                    if poly.shape[0] < 4:
                        print('poly:', poly.shape, np.array(inst['points']).shape)
                        continue
                    poly = poly.reshape(-1)
                    pt1 = (int(poly[0]), int(poly[1]))
                    pt2 = (int(poly[2]), int(poly[3]))
                    pt3 = (int(poly[4]), int(poly[5]))
                    pt4 = (int(poly[6]), int(poly[7]))

                    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

                    angle = 0

                    if edge1 > edge2:

                        width = edge1
                        height = edge2
                        if pt1[0] - pt2[0] != 0:
                            angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                        else:
                            angle = 90.0
                    elif edge2 >= edge1:
                        width = edge2
                        height = edge1
                        # print pt2[0], pt3[0]
                        if pt2[0] - pt3[0] != 0:
                            angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                        else:
                            angle = 90.0
                    if angle < -45.0:
                        angle = angle + 180

                    x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                    y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

                    if height * width < 8 * 8 and mode == "train":
                        continue

                    if "confidence" in word:
                        if "Low" in word["confidence"]:
                            hard_boxes.append([x_ctr, y_ctr, width, height, angle])
                    else:
                        easy_boxes.append([x_ctr, y_ctr, width, height, angle])

                    # if "confidence" in word:
                    #     if "Low" in word["confidence"]:
                    #         im = draw_boxes(im, word_box, (0, 0, 255), 3)
                    # else:
                    #     im = draw_boxes(im, word_box, (0, 0, 255))

            # cv2.imshow("wins", cv2.resize(im, (int(im.shape[1] / 2), int(im.shape[0] / 2))))
            # cv2.waitKey(0)

        # boxes = []
        boxes.extend(easy_boxes)
        boxes.extend(hard_boxes[0: int(len(hard_boxes))])

        len_of_bboxes = len(boxes)
        gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
        gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32)  # text or non-text
        seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

        for idx in range(len(boxes)):
            gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
            gt_classes[idx] = 1  # cls_text
            overlaps[idx, 1] = 1.0  # cls_text
            seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])

        # img_pil = vis_image(img_pil, gt_boxes)
        # img_pil.save('gt_LSVT.jpg', 'jpeg')
        # break
        max_overlaps = overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = overlaps.argmax(axis=1)
        if gt_boxes.shape[0] <= 0:
            continue
        # print('gt_boxes:', gt_boxes)
        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': impath,
            'boxes': gt_boxes,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': recRes["height"],
            'width': recRes["width"],
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    print('Saving pkls...')
    pkl_f = open(cache_file, 'wb')
    pickle.dump(im_infos, pkl_f)
    pkl_f.close()
    print('done')
    return im_infos


def get_docVQA_formal(mode, dataset_dir):

    cache_file = './data_cache/docVQA_det_' + mode + '_1_0.pkl'
    if os.path.isfile(cache_file):
        print('dataset cache found, loading from it...')
        im_infos = pickle.load(open(cache_file, 'rb'))
        print('load done')
        return im_infos

    # im_dir = os.path.join(dataset_dir, mode)
    # qa_file = os.path.join(dataset_dir, mode, mode + "_v1.0.json")

    # qa_json = eval(open(qa_file, "r", encoding="utf-8").read())

    # ocr_anno_dir = os.path.join(dataset_dir, mode, "ocr_results/")

    gt_dir = os.path.join(dataset_dir, mode, "ocr_results/")
    im_dir = os.path.join(dataset_dir, mode, "documents")

    gt_flist = os.listdir(gt_dir)

    print("gt_dir:", gt_dir, len(gt_flist))

    assert len(gt_flist) > 0, "False"

    gt_flist.sort()

    picked_gt_flist = []
    im_infos = []
    # Total in 4331
    # if mode == "train":
    #     picked_gt_flist = gt_flist[:4000]
    # elif mode == "validation":
    #     picked_gt_flist = gt_flist[4000:]
    # elif mode == "whole":
    #     picked_gt_flist = gt_flist

    flen = len(gt_flist)

    cnt = 0

    for gt_json in gt_flist:

        gt_path = os.path.join(gt_dir, gt_json)

        imprefix = gt_json.split(".")[0]
        impath = os.path.join(im_dir, imprefix + ".png")
        # im = cv2.imread(impath)
        cnt += 1
        if cnt % 100 == 0:
            print(impath, str(cnt) + "/" + str(flen))

        json_dict = json.load(open(gt_path, "r"))

        # print("json_dict:", json_dict.keys())

        boxes = []
        easy_boxes = []
        hard_boxes = []

        for recRes in json_dict["recognitionResults"]:

            # print("json_dict:", recRes)
            # print()

            for line in recRes["lines"]:

                # print("line:", line)

                line_box = np.array(line["boundingBox"])
                line_text = line["text"]
                # print("linekey:", line.keys())
                # im = draw_boxes(im, line_box, (255, 0, 0))

                for word in line["words"]:
                    # print("wordkey:", word.keys())
                    word_box = np.array(word["boundingBox"])
                    word_text = word["text"]

                    poly = word_box

                    if poly.shape[0] < 4:
                        print('poly:', poly.shape, np.array(inst['points']).shape)
                        continue
                    poly = poly.reshape(-1)
                    pt1 = (int(poly[0]), int(poly[1]))
                    pt2 = (int(poly[2]), int(poly[3]))
                    pt3 = (int(poly[4]), int(poly[5]))
                    pt4 = (int(poly[6]), int(poly[7]))

                    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

                    angle = 0

                    if edge1 > edge2:

                        width = edge1
                        height = edge2
                        if pt1[0] - pt2[0] != 0:
                            angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                        else:
                            angle = 90.0
                    elif edge2 >= edge1:
                        width = edge2
                        height = edge1
                        # print pt2[0], pt3[0]
                        if pt2[0] - pt3[0] != 0:
                            angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                        else:
                            angle = 90.0
                    if angle < -45.0:
                        angle = angle + 180

                    x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                    y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

                    if height * width < 8 * 8 and mode == "train":
                        continue

                    if "confidence" in word:
                        if "Low" in word["confidence"]:
                            hard_boxes.append([x_ctr, y_ctr, width, height, angle])
                    else:
                        easy_boxes.append([x_ctr, y_ctr, width, height, angle])

                    # if "confidence" in word:
                    #     if "Low" in word["confidence"]:
                    #         im = draw_boxes(im, word_box, (0, 0, 255), 3)
                    # else:
                    #     im = draw_boxes(im, word_box, (0, 0, 255))

            # cv2.imshow("wins", cv2.resize(im, (int(im.shape[1] / 2), int(im.shape[0] / 2))))
            # cv2.waitKey(0)

        # boxes = []
        boxes.extend(easy_boxes)
        boxes.extend(hard_boxes[0: int(len(hard_boxes))])

        len_of_bboxes = len(boxes)
        gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
        gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32)  # text or non-text
        seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

        for idx in range(len(boxes)):
            gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
            gt_classes[idx] = 1  # cls_text
            overlaps[idx, 1] = 1.0  # cls_text
            seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])

        # img_pil = vis_image(img_pil, gt_boxes)
        # img_pil.save('gt_LSVT.jpg', 'jpeg')
        # break
        max_overlaps = overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = overlaps.argmax(axis=1)
        if gt_boxes.shape[0] <= 0:
            continue
        # print('gt_boxes:', gt_boxes)
        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': impath,
            'boxes': gt_boxes,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': recRes["height"],
            'width': recRes["width"],
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    print('Saving pkls...')
    pkl_f = open(cache_file, 'wb')
    pickle.dump(im_infos, pkl_f)
    pkl_f.close()
    print('done')
    return im_infos


def get_doc_table_det(mode, dataset_dir):

    gt_json_file = os.path.join(dataset_dir, joint_gt.json)
    im_dir = os.path.join(dataset_dir, "images")

    json_f = open(gt_json_file, 'r')
    gt_annos = json.load(json_f)

    im_infos = []

    cache_file = "./data_cache/table_det_" + mode + ".pkl"
    if os.path.isfile(cache_file):
        print("cache_file found in ", cache_file, "loading...")
        cf = open(cache_file, "rb")
        return pickle.load(cf)

    for image_name in gt_annos:
        impath = os.path.join(im_dir, image_name)
        im = cv2.imread(impath)

        annos = gt_annos[image_name]
        boxes = []
        for line_eles in annos:
            l, t, r, b = line_eles[1:5]
            poly = np.array([l, t, r, t, r, b, l, b]).astype(np.int32)

            x_ctr, y_ctr, width, height, angle = poly2rbox_single(poly)

            boxes.append([x_ctr, y_ctr, width, height, angle])

        len_of_bboxes = len(boxes)
        gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
        gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32)  # text or non-text
        seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

        for idx in range(len(boxes)):
            gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
            gt_classes[idx] = 1  # cls_text
            overlaps[idx, 1] = 1.0  # cls_text
            seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])

        # img_pil = vis_image(img_pil, gt_boxes)
        # img_pil.save('gt_LSVT.jpg', 'jpeg')
        # break
        max_overlaps = overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = overlaps.argmax(axis=1)
        if gt_boxes.shape[0] <= 0:
            continue
        # print('gt_boxes:', gt_boxes)
        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': impath,
            'boxes': gt_boxes,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': im.shape[0],
            'width': im.shape[1],
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    cf = open(cache_file, "wb")
    pickle.dump(im_infos, cf)
    cf.close()
    print("pickle file save done")

    return im_infos



DATASET = {
    'IC13':get_ICDAR2013,
    'IC15':get_ICDAR2015_RRC_PICK_TRAIN,
    'IC17mlt':get_ICDAR2017_mlt,
    'LSVT':get_ICDAR_LSVT_full,
    'ArT':get_ICDAR_ArT,
    'ReCTs':get_ICDAR_ReCTs_full,
    "Syn800K": get_Syn_800K_with_words,
    "docVQA": get_docVQA,
    "docVQA_formal": get_docVQA_formal,
    "table_det": get_doc_table_det
}


_DEBUG = False
class RotationDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "text"
    )

    def __init__(self, database, use_difficult=False, transforms=None):
        # database:{dataset_name, dataset_dir}

        self.transforms = transforms

        self.annobase = []

        for dataset_name in database:
            if dataset_name in DATASET:
                self.annobase.extend(DATASET[dataset_name]('train', database[dataset_name]))
                if dataset_name == "LSVT":
                    self.annobase.extend(DATASET[dataset_name]('val', database[dataset_name]))

        # Do shuffle
        random.shuffle(self.annobase)

        print('DATASET: Total samples from:', database.keys(), len(self.annobase))

        self.ids = [anno['image'][:-4] for anno in self.annobase]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = RotationDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        # self.mixup = T.MixUp(mix_ratio=0.1)
        self.num_samples = len(self.annobase)

    def __getitem__(self, index):

        if _DEBUG:
            index = 5

        # img_id = self.ids[index]

        im_path = self.annobase[index]['image'] # index
        img = Image.open(im_path).convert("RGB")
        # im = cv2.imread(im_path)
        anno = self.annobase[index] # index
        target = RBoxList(torch.from_numpy(anno["boxes"]), (anno['width'], anno['height']), mode="xywha")
        target.add_field("labels", torch.from_numpy(anno["gt_classes"]))
        target.add_field("difficult", torch.Tensor([0 for i in range(len(anno["gt_classes"]))]))

        target = target.clip_to_image(remove_empty=True)
        # print('target:', target, im_path)
        if self.transforms is not None:
            # off = int(self.num_samples * np.random.rand())
            # mix_index = (off + index) % self.num_samples
            # img_mix = Image.open(self.annobase[mix_index]['image']).convert("RGB")
            # img, target = self.mixup(img, img_mix, target)
            img, target = self.transforms(img, target)

            if target is None:
                return self[(index + 1) % self.__len__()] #index + 1

        if _DEBUG:
            if not target is None:
                self.show_boxes(img, target)

        return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):

        return {"height": self.annobase[index]['height'], "width": self.annobase[index]['width']}

    def map_class_id_to_class_name(self, class_id):
        return RotationDataset.CLASSES[class_id]

    def show_boxes(self, img, target):
        bbox_np = target.bbox.data.cpu().numpy()
        # print('image shape:', img.size())
        np_img = np.transpose(np.uint8(img.data.cpu().numpy()), (1, 2, 0))
        img_pil = Image.fromarray(np_img)
        # print('bbox_np:', bbox_np)
        draw_img = vis_image(img_pil, bbox_np)
        draw_img.save('gt_show.jpg', 'jpeg')
        # print('Sleep for show...')
        # time.sleep(2)

if __name__ == '__main__':
    get_ICDAR_LSVT_full('train', '../datasets/LSVT/')