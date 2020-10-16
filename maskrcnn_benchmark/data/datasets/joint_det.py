import os
import pickle
import torch
import torch.utils.data
import torchvision
from PIL import Image
import sys
import numpy as np
import time
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import RBoxList, BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.utils.visualize import vis_image
from maskrcnn_benchmark.data.transforms import transforms as T

from maskrcnn_benchmark.data.datasets.coco_text import coco_text as ct
import cv2

from scipy import io as sio
import re
import json
import random
import math

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
        cache_pkl = './data_cache/Syn_800K_training_E2E.pkl'
    else:
        cache_pkl = './data_cache/Syn_800K_testing_E2E.pkl'

    if os.path.isfile(cache_pkl):
        return pickle.load(open(cache_pkl, 'rb'))

    pro_cnt = 0

    for i in range(sam_size):
        txts = txt_annos[i]
        im_path = os.path.join(dataset_dir, im_names[i][0])
        word_boxes = wordBBs[i]

        pro_cnt += 1
        if pro_cnt % 2000 == 0:
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

                        if height * width * (800 / float(im.shape[0])) < 16 * 32 and mode == "train":
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


def get_Syn_90Klex_with_words(mode, dataset_dir):
    # if mode == 'train':
    #    image_dir = os.path.join(dataset_dir, 'image_9000/')
    # gt_dir = os.path.join(dataset_dir, 'txt_9000/')

    # ./ICPR_dataset/update_ICPR_text_train_part1_20180316/train_1000/
    # else:
    #    image_dir = os.path.join(dataset_dir, 'image_1000/')
    # gt_dir = os.path.join(dataset_dir, 'txt_1000/')
    '''
    word2vec_mat = '../selected_smaller_dic.mat'
    mat_data = sio.loadmat(word2vec_mat)
    all_words = mat_data['selected_vocab']
    all_vecs = mat_data['selected_dict']

    w2v_dict = {}
    print('Building w2v dictionary...')
    for i in range(len(all_words)):
        w2v_dict[all_words[i][0][0]] = all_vecs[i]
    print('done')
    '''
    # mat_file = os.path.join(dataset_dir, 'gt.mat')
    # print('mat_file:', mat_file)
    # mat_f = sio.loadmat(mat_file)

    # wordBBs = mat_f['wordBB'][0]
    # txt_annos = mat_f['txt'][0]
    # im_names = mat_f['imnames'][0]

    sam_size = 200000

    # image_list = os.listdir(image_dir)
    # image_list.sort()
    im_infos = []

    if mode == 'train':
        cache_pkl = './data_cache/Syn_90Klex_training.pkl'
    else:
        cache_pkl = './data_cache/Syn_90Klex_testing.pkl'

    if os.path.isfile(cache_pkl):
        return pickle.load(open(cache_pkl, 'rb'))

    pro_cnt = 0

    case = ['syn_img', 'syn_img_lower']
    sub_folder = [i for i in range(10)]

    for i in range(sam_size):
        # txts = txt_annos[i]
        # im_path = os.path.join(dataset_dir, im_names[i][0])
        # word_boxes = wordBBs[i]

        case_dir = case[int(i / 100000)]
        im_idx = str(int(i % 100000))
        sub_folder = str(int(int(im_idx) / 10000))

        im_path = os.path.join(dataset_dir, case_dir, 'Img', sub_folder, im_idx + '.jpg')
        txt_path = os.path.join(dataset_dir, case_dir, 'Txt', sub_folder, im_idx + '.txt')

        pro_cnt += 1
        if pro_cnt % 2000 == 0:
            print('processed image:', str(pro_cnt) + '/' + str(sam_size))

        cnt = 0
        # print('word_boxes:', word_boxes.shape)
        im = cv2.imread(im_path)

        # if len(word_boxes.shape) < 3:
        #    word_boxes = np.expand_dims(word_boxes, -1)
        words = []
        boxes = []
        word_vecs = []

        txts = open(txt_path, 'r').readlines()

        for txt in txts:
            txtsp = txt.split(',')
            # for line in txtsp:
            #    line = line.replace('\n', '').replace('\n', '').replace('\r', '').replace('\t', '').split(' ')
            #    #print('line:', line)
            #    for w in line:

            # w = w
            if len(txtsp) > 0:
                gt_ind = np.array(txtsp[:8],
                                  dtype=np.int32)  # np.transpose(np.array(word_boxes[:, :, cnt], dtype=np.int32), (1, 0)).reshape(8)
                w = txtsp[-1].replace('\n', '')
                # print(im_path, gt_ind, w)

                cnt += 1

                # cv2.line(im, (gt_ind[0], gt_ind[1]), (gt_ind[2], gt_ind[3]), (0, 0, 255), 3)

                # cv2.line(im, (gt_ind[2], gt_ind[3]), (gt_ind[4], gt_ind[5]), (0, 0, 255), 3)
                # cv2.line(im, (gt_ind[4], gt_ind[5]), (gt_ind[6], gt_ind[7]), (0, 0, 255), 3)
                # cv2.line(im, (gt_ind[6], gt_ind[7]), (gt_ind[0], gt_ind[1]), (0, 0, 255), 3)
                # cv2.putText(im, w, (gt_ind[0], gt_ind[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 122), 2)

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

                if height * width * (800 / float(im.shape[0])) < 16 * 16 * len(w) and mode == "train":
                    continue
                if x_ctr >= im.shape[1] or x_ctr < 0 or y_ctr >= im.shape[0] or y_ctr < 0:
                    continue

                # com_num = re.compile('[0-9]+')
                # com_prices = re.compile('[$￥€£]+')

                # match_num = re.findall(com_num, w)
                # match_prices = re.findall(com_prices, w)
                # choices: original, prices, others
                words.append(w)
                '''
                w = w.lower()
                if w in w2v_dict:
                    # print(w)
                    word_vecs.append(w2v_dict[w])
                elif match_prices and match_num:
                    # print('price')
                    word_vecs.append(w2v_dict['price'])
                elif match_num and not match_prices:
                    # print('ten')
                    word_vecs.append(w2v_dict['ten'])
                else:
                    # print(im_path, w)
                    # print('0')
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

        # cv2.imshow('win', im)
        # cv2.waitKey(0)

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
            #'gt_wordvec': np.array(word_vecs),
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    f_save_pkl = open(cache_pkl, 'wb')
    pickle.dump(im_infos, f_save_pkl)
    f_save_pkl.close()
    print("Save pickle done.")
    return im_infos


def get_ICDAR2015_RRC_PICK_TRAIN_with_words(mode, dataset_dir):
    # dir_path = "/home/shiki-alice/Downloads/ICDAR2015/ch4_training_images/"
    img_file_type = "jpg"

    # gt_dir = "/home/shiki-alice/Downloads/ICDAR2015/ch4_training_localization_transcription_gt/"

    image_dir = os.path.join(dataset_dir, 'ch4_training_images/')
    gt_dir = os.path.join(dataset_dir, 'ch4_training_localization_transcription_gt/')

    image_list = os.listdir(image_dir)
    image_list.sort()
    im_infos = []
    '''
    word2vec_mat = 'selected_smaller_dic.mat'
    mat_data = sio.loadmat(word2vec_mat)
    all_words = mat_data['selected_vocab']
    all_vecs = mat_data['selected_dict']

    w2v_dict = {}
    print('Building w2v dictionary...')
    for i in range(len(all_words)):
        w2v_dict[all_words[i][0][0]] = all_vecs[i]
    print('done')
    '''
    cache_file = './data_cache/IC15_E2E_train.pkl'
    if os.path.isfile(cache_file):
        return pickle.load(open(cache_file, 'rb'))

    for image in image_list:

        prefix = image[:-4]
        img_name = os.path.join(image_dir, image)
        gt_name = os.path.join(gt_dir, 'gt_' + prefix + '.txt')

        # img_name = dir_path + img_list[idx]
        # gt_name = gt_dir + gt_list[idx]

        easy_boxes = []
        easy_words = []
        hard_boxes = []

        boxes = []
        # print gt_name
        gt_obj = open(gt_name, 'r')

        gt_txt = gt_obj.read()

        gt_split = gt_txt.split('\n')

        img = cv2.imread(img_name)

        word_vecs = []

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

                w = gt_ind[8].replace('\n', '').replace('\r', '')
                easy_boxes.append([x_ctr, y_ctr, width, height, angle])
                easy_words.append(w)

                #com_num = re.compile('[0-9]+')
                #com_prices = re.compile('[$￥€£]+')

                #match_num = re.findall(com_num, w)
                #match_prices = re.findall(com_prices, w)

                # choices: original, prices, others
                # words.append(w)
                '''
                w = w.lower()
                if w in w2v_dict:
                    word_vecs.append(w2v_dict[w.lower()])
                elif match_prices and match_num:
                    word_vecs.append(w2v_dict['price'])
                elif match_num and not match_prices:
                    word_vecs.append(w2v_dict['ten'])
                else:
                    print(img_name, w)
                    word_vecs.append(np.zeros(100, dtype=np.float32) + 1e-10)
                '''
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

                hard_boxes.append([x_ctr, y_ctr, width, height, angle])

        boxes.extend(easy_boxes)

        # boxes.extend(hard_boxes[0 : int(len(hard_boxes) / 3)])

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
        if gt_boxes.shape[0] <= 0:
            continue
        # print('gt_boxes:', gt_boxes)
        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': img_name,
            'boxes': gt_boxes,
            'gt_words': easy_words,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': img.shape[0],
            'width': img.shape[1],
            # 'gt_wordvec': np.array(word_vecs),
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    ca_f = open(cache_file, 'wb')
    pickle.dump(im_infos, ca_f)
    print('IC15 pkl save done')

    return im_infos


def get_ICDAR2013_with_words(mode, dataset_dir):
    DATASET_DIR = dataset_dir

    img_dir = "/ch2_training_images/"
    gt_dir = "/ch2_training_localization_transcription_gt"

    # gt_list = []
    # img_list = []

    im_infos = []
    image_dir = DATASET_DIR + img_dir
    gt_file_list = os.listdir(image_dir)

    if mode == 'train':
        cache_pkl = 'data_cache/IC13_training_e2e.pkl'
    '''
    word2vec_mat = 'selected_smaller_dic.mat'
    mat_data = sio.loadmat(word2vec_mat)
    all_words = mat_data['selected_vocab']
    all_vecs = mat_data['selected_dict']

    w2v_dict = {}
    print('Building w2v dictionary...')
    for i in range(len(all_words)):
        w2v_dict[all_words[i][0][0]] = all_vecs[i]
    print('done')
    '''
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
        gt_words = []
        boxes = []
        word_vecs = []
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
                if '###' in words:
                    continue

                # com_num = re.compile('[0-9]+')
                # com_prices = re.compile('[$￥€£]+')

                # match_num = re.findall(com_num, words)
                # match_prices = re.findall(com_prices, words)

                # choices: original, prices, others
                # words.append(w)
                gt_words.append(words)
                '''
                words = words.lower()
                if words in w2v_dict:
                    word_vecs.append(w2v_dict[words.lower()])
                elif match_prices and match_num:
                    word_vecs.append(w2v_dict['price'])
                elif match_num and not match_prices:
                    word_vecs.append(w2v_dict['ten'])
                else:
                    print(im_path, words)
                    word_vecs.append(np.zeros(100, dtype=np.float32) + 1e-10)
                # return to width, height
                '''
                boxes.append([x_ctr, y_ctr, width, height, angle, words])

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
            'gt_words': gt_words,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': im.shape[0],
            'width': im.shape[1],
            #'gt_wordvec': np.array(word_vecs),
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    f_save_pkl = open(cache_pkl, 'wb')
    pickle.dump(im_infos, f_save_pkl)
    f_save_pkl.close()
    print("Save pickle done.")

    return im_infos


def get_ICDAR_LSVT_full_with_words(mode, dataset_dir):

    assert mode in ['train', 'val', 'full'], 'mode not in ' + str(['train', 'val', 'full'])

    data_split = {
        'val':[0, 3000],
        'train':[3000, 30000],
        'full':[0, 30000]
    }

    vis = False

    cache_file = './data_cache/LSVT_' + mode + '_E2E.pkl'
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

        words = []

        print(str(imnum) + '/' + str(data_split[mode][0] + num_samples), impath)

        for i in range(inst_num):
            inst = gt_anno[i]
            poly = np.array(inst['points'])
            word = inst['transcription']
            illegibility = inst['illegibility']

            if illegibility:
                continue
            if len(word) >= 35:
                continue
            if len(word) < 1:
                continue
            # print(word)
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

            #
            #    hard_boxes.append([x_ctr, y_ctr, width, height, angle])
            #else:
            easy_boxes.append([x_ctr, y_ctr, width, height, angle])
            words.append(word)
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
            'gt_words': words,
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


def get_ICDAR_LSVT_weak_with_words(mode, dataset_dir):

    assert mode in ['train'], 'mode not in ' + str(['train'])

    data_split = {
        'train':[0, 80000],
    }

    vis = False

    cache_file = './data_cache/LSVT_weak_' + mode + '_E2E.pkl'
    if os.path.isfile(cache_file):
        print('dataset cache found, loading from it...')
        im_infos = pickle.load(open(cache_file, 'rb'))
        print('load done')
        return im_infos

    im_codes = range(data_split[mode][0], data_split[mode][1])
    gt_json = os.path.join(dataset_dir, 'train_weak_labels.json')
    print('gt_json:', gt_json)
    gt_dict = json.load(open(gt_json, 'r'))

    im_infos = []

    num_samples = data_split[mode][1] - data_split[mode][0]

    for imnum in im_codes:
        forder = int(imnum / 40000)
        imfolder = os.path.join(dataset_dir, 'train_weak_images_'+str(forder))
        impath = os.path.join(imfolder, 'gt_' + str(imnum) + '.jpg')
        gt_code = 'gt_' + str(imnum)
        gt_anno = gt_dict[gt_code]

        inst_num = len(gt_anno)

        im = cv2.imread(impath)

        easy_boxes = []
        hard_boxes = []

        words = []

        print(str(imnum) + '/' + str(data_split[mode][0] + num_samples), impath)

        if im is None:
            print(impath, im)
            continue

        for i in range(inst_num):
            inst = gt_anno[i]
            word = inst['transcription']
            #illegibility = inst['illegibility']

            if len(word) >= 35:
                continue
            if len(word) < 1:
                continue

            words.append(word)
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
        if len(words) <= 0:
            continue
        # print('gt_boxes:', gt_boxes)
        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': impath,
            'boxes': gt_boxes,
            'flipped': False,
            'gt_words': words,
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




def get_docVQA_with_words(mode, dataset_dir):

    cache_file = './data_cache/docVQA_e2e_' + mode + '.pkl'
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

        words = []
        easy_words = []
        hard_words = []

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
                            hard_words.append(word_text)
                        else:
                            easy_boxes.append([x_ctr, y_ctr, width, height, angle])
                            easy_words.append(word_text)
                    else:
                        easy_boxes.append([x_ctr, y_ctr, width, height, angle])
                        easy_words.append(word_text)
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

        words.extend(easy_words)
        words.extend(hard_words[0: int(len(hard_boxes))])

        len_of_bboxes = len(boxes)
        gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
        gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32)  # text or non-text
        seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

        for idx in range(len(boxes)):

            # print("gt_boxes:", boxes[idx])

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
            'gt_words': words,
            'rotated': True
        }
        im_infos.append(im_info)

    print('Saving pkls...')
    pkl_f = open(cache_file, 'wb')
    pickle.dump(im_infos, pkl_f)
    pkl_f.close()
    print('done')
    return im_infos

def get_ICDAR2017_mlt_with_words(mode, dataset_dir, alphabet=""):
    DATASET_DIR = dataset_dir
    task = 'double_class'
    prefetched = True if os.path.isfile('./data_cache/ICDAR2017_' + mode + '_cache_E2E.pkl') else False
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

                gt_candidate_path = DATASET_DIR + "ch8_training_localization_transcription_gt/" + 'gt_img_' + str(
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

                easy_words = []

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

                            ori_word = gt_ind[9].replace("\r", "").replace("\n", "")
                            get_word = True
                            for ch in ori_word:
                                if ch not in alphabet:
                                    get_word = False
                                    break
                            if get_word and len(ori_word) > 0:
                                easy_boxes.append([x_ctr, y_ctr, width, height, angle, gt_ind[8]])
                                easy_words.append(ori_word)
                            else:
                                print("ori_word:", ori_word)
                boxes.extend(easy_boxes)

                # all_len = len(easy_boxes) + int(len(hard_boxes))
                # if all_len <= 50:
                # boxes.extend(hard_boxes[0: int(len(hard_boxes) * 0.2)])
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
                    "gt_words": easy_words,
                    'rotated': True
                }
                im_infos.append(im_info)

            f_save_pkl = open('./data_cache/ICDAR2017_' + mode + '_cache_E2E.pkl', 'wb')
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

                gt_candidate_path = DATASET_DIR + "ch8_validation_localization_transcription_gt/" + 'gt_img_' + str(
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

            f_save_pkl = open('./data_cache/ICDAR2017_' + mode + '_cache_E2E.pkl', 'wb')
            pickle.dump(im_infos, f_save_pkl)
            f_save_pkl.close()
            print ("Save pickle done.")
    else:
        if mode == "train":
            f_pkl = open('./data_cache/ICDAR2017_' + mode + '_cache_E2E.pkl', 'rb')
            im_infos = pickle.load(f_pkl)
        if mode == "validation":
            f_pkl = open('./data_cache/ICDAR2017_' + mode + '_cache_E2E.pkl', 'rb')
            im_infos = pickle.load(f_pkl)
    return im_infos


def get_cocotext_with_words(mode, dataset_dir, alphabet=""):

    cache_file = './data_cache/cocotext_e2e_' + mode + '.pkl'

    if os.path.isfile(cache_file):
        print('dataset cache found, loading from it...')
        im_infos = pickle.load(open(cache_file, 'rb'))
        print('load done')
        return im_infos

    im_infos = []

    gt_file = os.path.join(dataset_dir, "cocotext.v2.json")
    im_dir = os.path.join(dataset_dir, "train2014")

    # annobase = json.load(open(gt_file, "r"))

    ct_base = ct.COCO_Text(gt_file)

    # print("annobase:", ct_base.info())

    imgs = ct_base.getImgIds(imgIds=ct_base.train,
                        catIds=[('legibility', 'legible')])

    # anns = ct_base.getAnnIds(imgIds=ct_base.train,
    #                     catIds=[('legibility', 'legible'), ('class', 'machine printed')],
    #                    areaRng=[0, 200])

    # img_paths = ct_base.loadImgs(imgs[:100])

    # print("anns:", len(anns), anns[:100])
    # print("imgs:", len(imgs), imgs[:100], img_paths)

    all_im = len(imgs)
    cnt = 0

    for im_id in imgs:

        if cnt % 100 == 0:
            print("processing:", cnt, "/", all_im)
        cnt += 1
        img_info = ct_base.loadImgs(im_id)[0]
        annIds = ct_base.getAnnIds(imgIds=im_id)
        anns = ct_base.loadAnns(annIds)

        # print("img_info:", img_info)

        impath = os.path.join(im_dir, img_info["file_name"])

        im_height = img_info["height"]
        im_width = img_info["width"]

        # img = cv2.imread(impath)
        # img = Image.fromarray(img)

        boxes = []
        words = []

        for ann in anns:
            rpoly = np.array(ann["mask"]).reshape(-1, 2).astype(np.int32)
            word = ann["utf8_string"]
            lang = ann["language"]

            # print("rpoly:", rpoly, ann)
            if rpoly.shape[0] != 4:
                rect = cv2.minAreaRect(rpoly)
                poly = np.array(cv2.boxPoints(rect), np.int)

            else:
                poly = rpoly
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
            # easy_polys = []
            # hard_polys = []

            if height * width < 8 * 8 and mode == "train":
                continue
            # return to width, height

            get_word = True
            for ch in word:
                if ch not in alphabet:
                    get_word = False
                    break

            if '###' in word or lang not in ["english"] or len(word) < 1 or not get_word:
                continue

            boxes.append([x_ctr, y_ctr, width, height, angle])
            words.append(word)

        # img = vis_image(img, boxes)
        # cv2.imshow("win", np.array(img))
        # cv2.waitKey(0)
        # img.save(img_info["file_name"], "JPEG")
        # print(boxes, words)

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
        if len(words) <= 0:
            continue
        # print('gt_boxes:', gt_boxes)
        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': impath,
            'boxes': gt_boxes,
            'flipped': False,
            'gt_words': words,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': im_height,
            'width': im_width,
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


DATASET = {
    'IC13':get_ICDAR2013_with_words,
    'IC15':get_ICDAR2015_RRC_PICK_TRAIN_with_words,
    "IC17mlt": get_ICDAR2017_mlt_with_words,
    '90Klex':get_Syn_90Klex_with_words,
    'Syn800K':get_Syn_800K_with_words,
    'LSVT':get_ICDAR_LSVT_full_with_words,
    'LSVT_weak':get_ICDAR_LSVT_weak_with_words,
    'docVQA': get_docVQA_with_words,
    "coco-text": get_cocotext_with_words
}


_DEBUG = False
from maskrcnn_benchmark.utils.rec_utils import StrLabelConverter

class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

            ids_to_remove = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anno = self.coco.loadAnns(ann_ids)
                if all(
                    any(o <= 1 for o in obj["bbox"][2:])
                    for obj in anno
                    if obj["iscrowd"] == 0
                ):
                    ids_to_remove.append(img_id)

            self.ids = [
                img_id for img_id in self.ids if img_id not in ids_to_remove
            ]

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data



class JointDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "text"
    )

    COCO_CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(self,
                 database,
                 alphabet_f=None,
                 use_difficult=False,
                 transforms=None,
                 coco_rootdir="./"):
        # database:{dataset_name, dataset_dir}

        self.transforms = transforms

        ########################### Text Processing ###########################

        self.annobase = []

        dataset_list = list(database.keys())
        dataset_list.sort()

        self.alphabet = ""

        if os.path.isfile(alphabet_f):
            print("alphabet_f assigned:", alphabet_f)
            self.key_profile = alphabet_f
            print(self.key_profile + ' found, loading...')
            self.alphabet = open(self.key_profile, 'r').read()
        else:
            print("Provided alphabet_f not found, remake it...")
            self.key_profile = None

        for dataset_name in dataset_list:

            if dataset_name in DATASET:
                if dataset_name in ["IC17mlt", "coco-text"]:
                    dataset = DATASET[dataset_name]('train', database[dataset_name], self.alphabet)
                else:
                    dataset = DATASET[dataset_name]('train', database[dataset_name])
                # do shuffle
                # random.shuffle(dataset)
                # self.annobase.append(dataset)
                self.annobase.extend(dataset)

        random.shuffle(self.annobase)

        # if os.path.isfile(self.key_profile):

        if self.key_profile is None:

            self.key_profile = './data_cache/alphabet_'
            for dataset_name in dataset_list:
                if dataset_name != 'LSVT_weak':
                    self.key_profile += dataset_name + '_'
            self.key_profile += 'pro.txt'

            for anno in self.annobase:
                words = anno['gt_words']
                for word in words:
                    for ch in word:
                        if ch not in self.alphabet:
                            self.alphabet += ch
            dic_temp = list(self.alphabet)
            dic_temp.sort()
            dic_str = ''
            for i in range(len(dic_temp)):
                dic_str += dic_temp[i]
            self.alphabet = dic_str

            print('Saving alphabet into ' + self.key_profile)
            ca_f = open(self.key_profile, 'w')
            ca_f.write(self.alphabet)
            ca_f.flush()
            ca_f.close()

            # self.annobase_cat

        self.ids = [anno['image'][:-4] for anno in self.annobase]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        # del self.annobase_cat

        cls = JointDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

        self.wk_converter = StrLabelConverter(self.alphabet)

        self.database_num = len(self.annobase)
        ###################################################################

        ########################### Coco Object ###########################

        json_file = os.path.join(coco_rootdir, "annotations/instances_train2014.json")
        gen_img_dir = os.path.join(coco_rootdir, "train2014")
        cocoDet = torchvision.datasets.coco.CocoDetection(gen_img_dir, json_file)
        self.coco = cocoDet.coco
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        ###################################################################
        print("Database:", dataset_list, len(self.annobase))

        self.mixup = T.MixUp()

    def get_single_item(self, index):

        anno = self.annobase[index % self.__len__()]
        # [
        #     int(index / self.database_num) % len(self.annobase[index % self.database_num])]
        im_path = anno['image']
        img = Image.open(im_path).convert("RGB")
        # print('im_path:', im_path)
        text, text_len = self.wk_converter.encode(anno['gt_words'])

        text_label_split = []

        off_cnt = 0

        mx_len = np.max(text_len)
        word_num = len(text_len)

        for i in range(len(text_len)):
            text_label_split.append(text[off_cnt:off_cnt + text_len[i]])
            off_cnt += text_len[i]

        padding_words = np.zeros((word_num, mx_len))
        for i in range(word_num):
            padding_words[i][:text_len[i]] = text_label_split[i]

        if anno["boxes"].shape[0] > 0:
            txt_target = RBoxList(torch.from_numpy(anno["boxes"]), (anno['width'], anno['height']), mode="xywha")
            txt_target.add_field("labels", torch.from_numpy(anno["gt_classes"]))
            txt_target.add_field("difficult", torch.tensor([0 for i in range(len(anno["gt_classes"]))]))
            txt_target.add_field("words", torch.from_numpy(padding_words))
            txt_target.add_field("word_length", torch.tensor(text_len))
            txt_target = txt_target.clip_to_image(remove_empty=True)
        else:
            txt_target = None

        # Get coco target
        coco_imid = int(im_path.split("/")[-1].split(".")[0][-6:])
        annIds = self.coco.getAnnIds(imgIds=coco_imid)
        anno = self.coco.loadAnns(annIds)

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        gen_target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        gen_target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        gen_target.add_field("masks", masks)

        gen_target = gen_target.clip_to_image(remove_empty=True)

        # if self.transforms is not None:
        #     _img, gen_target = self.transforms(img, gen_target)

        if len(boxes) < 1:
            gen_target = None

        return img, txt_target, gen_target

    def __getitem__(self, index):

        if _DEBUG:
            index = 0
        '''
        crop_imlist = []
        crop_tarlist = []

        for i in range(4):
            img, target = self.get_single_item(index)
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            crop_imlist.append(img)
            crop_tarlist.append(target)

        img, target = self.mixup(crop_imlist, crop_tarlist)
        '''

        img, txt_target, gen_target = self.get_single_item(index)
        if self.transforms is not None:
            txt_img, txt_target = self.transforms(img, txt_target)
            gen_img, gen_target = self.transforms(img, gen_target)
            img = txt_img

        if _DEBUG and not txt_target is None and not gen_target is None:
            self.show_boxes(img, txt_target)
            self.show_coco_boxes(img, gen_target)

        if txt_target is None or gen_target is None:
            return self[(index + 1) % self.__len__()]

        return img, {"text": txt_target, "general": gen_target}, index

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        # anno = self.annobase[index % self.database_num][int(index / self.database_num) % len(self.annobase[index % self.database_num])]
        anno = self.annobase[index]
        return {"height": anno['height'], "width": anno['width']}

    def map_class_id_to_class_name(self, class_id):
        return JointDataset.CLASSES[class_id]

    def show_boxes(self, img, target):
        bbox_np = target.bbox.data.cpu().numpy()
        # print('image shape:', img.size())
        # np_img = np.transpose(np.uint8(img.data.cpu().numpy()), (1, 2, 0))
        img_pil = img # Image.fromarray(np_img)
        draw_img = vis_image(img_pil, bbox_np)
        draw_img.save('gt_show.jpg', 'jpeg')
        # print('Sleep for show...')
        # time.sleep(2)

    def show_coco_boxes(self, img, target):
        bbox_np = target.bbox.data.cpu().numpy()
        labels = target.get_field("labels")
        # np_img = np.transpose(np.uint8(img.data.cpu().numpy()), (1, 2, 0))
        img_pil = img #Image.fromarray(np_img)

        x, y, w, h = bbox_np[:, 0:1], bbox_np[:, 1:2], bbox_np[:, 2:3], bbox_np[:, 3:]

        l = x
        t = y
        r = w
        b = h
        print("labels:", labels.data.cpu().numpy())
        qboxes = np.concatenate([l, t, r, t, r, b, l, b], axis=-1)
        draw_img = vis_image(img_pil, qboxes, cls_prob=labels.data.cpu().numpy(), repre="qbox")
        draw_img.save('gt_coco_show.jpg', 'jpeg')

if __name__ == "__main__":
    # get_cocotext_with_words("train", "../hard_dataset/dataset/COCO2014/")

    jbase = JointDataset(
                     database={'coco-text': '../hard_dataset/dataset/COCO2014/'},
                     alphabet_f="./data_cache/alphabet_IC13_IC15_Syn800K_pro.txt",
                     use_difficult=False,
                     transforms=None,
                     coco_rootdir='../hard_dataset/dataset/COCO2014/'
                 )


    for i in range(100):
        img, target, index = jbase[i]
        print(target.keys())