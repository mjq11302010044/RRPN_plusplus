import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import numpy as np
from maskrcnn_benchmark.config import cfg

import torch.backends.cudnn as cudnn
cudnn.enabled = False


from demo.predictor import ICDARDemo, RRPNDemo
from maskrcnn_benchmark.utils.visualize import vis_image, write_result_ICDAR_RRPN2polys, zip_dir, vis_image_with_words, write_result_ICDAR_RRPN2polys_with_words
from PIL import Image
import time


config_file = "./configs/arpn_E2E/e2e_rrpn_R_50_C4_1x_test_AFPN_RT_LERB_Spotter.yaml" #e2e_rrpn_R_50_C4_1x_ICDAR13_15_trial_test.yaml

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
# cfg.freeze()
# cfg.MODEL.WEIGHT = 'models/IC-13-15-17-Trial/model_0155000.pth'

result_dir = os.path.join('results', config_file.split('/')[-1].split('.')[0], cfg.MODEL.WEIGHT.split('/')[-1].split('.')[0])

if cfg.TEST.MODE == "DET":
    result_dir = result_dir + "_" + cfg.TEST.MODE

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)


coco_demo = RRPNDemo(
    cfg,
    min_image_size=1440,
    confidence_threshold=0.1,
)

dataset_name = cfg.TEST.DATASET_NAME

testing_dataset = {
     'IC13': {
        'testing_image_dir': '../hard_space1/mjq/ICDAR13/Challenge2_Test_Task12_Images',
        'test_vocal_dir': '../hard_space1/mjq/ICDAR13/ch2_test_vocabularies_per_image',
        'off': [0, 233],
    },   
    'IC15': {
        'testing_image_dir': '../hard_space1/mjq/ICDAR15/ch4_test_images',
        'test_vocal_dir': '../hard_space1/mjq/ICDAR15/ch4_test_vocabularies_per_image',
        'off': [0, 500],
    },
    'IC17mlt_test':{
        'testing_image_dir':
            '../hard_space1/mjq/ICDAR17MLT/ch8_test_images/',
        'off': [0, 9100],
        "json_file": "train_task_1.json",
        # [4000, 4330],
        # 'gt_dir':"../dataset/ICDAR15/Challenge4_Test_Task4_GT"
    },
    'docVQA':{
        'testing_image_dir':
            '../dataset/docVQA/docvqa_train_v0.1/documents/',
        'off': [4000, 4330]# [4000, 4330],
        # 'gt_dir':"../dataset/ICDAR15/Challenge4_Test_Task4_GT"
    },
    'docVQA_TEST':{
        'testing_image_dir':
            '../dataset/docVQA/test/documents/',
        'off': [0, 1000000]# [4000, 4330],
        # 'gt_dir':"../dataset/ICDAR15/Challenge4_Test_Task4_GT"
    },
    'COCO-TEXT':{
            'testing_image_dir':
                '../hard_space1/mjq/coco_text/train2014/',
            'off': [0, 40000],
            # "json_file": "train_task_1.json"
            # [4000, 4330],
            # 'gt_dir':"../dataset/ICDAR15/Challenge4_Test_Task4_GT"
        }
}

image_dir = testing_dataset[dataset_name]['testing_image_dir']
vocab_dir = testing_dataset[dataset_name]['test_vocal_dir'] \
    if 'test_vocal_dir' in testing_dataset[dataset_name] else None

# load image and then run prediction
# image_dir = '../datasets/ICDAR13/Challenge2_Test_Task12_Images/'
imlist = os.listdir(image_dir)
alphabet = open(cfg.MODEL.ROI_REC_HEAD.ALPHABET).readlines()[0] + '-'

print('************* META INFO ***************')
print('config_file:', config_file)
print('result_dir:', result_dir)
print('image_dir:', image_dir)
print('weights:', cfg.MODEL.WEIGHT)
print('alphabet:', alphabet)
print('***************************************')

vis = True
vis_dir = "vis_E2E_" + dataset_name

if vis and not os.path.isdir(vis_dir):
    os.makedirs(vis_dir)

off_start, off_end = testing_dataset[dataset_name]["off"]

cnt = 0

imlist.sort()

if off_end > len(imlist):
    off_end = len(imlist)

num_images = off_end - off_start

for idx in range(off_start, off_end):

    image = imlist[idx]

    off = image.split('.')[0]

    impath = os.path.join(image_dir, image)

    vocabs = None

    if not vocab_dir is None:
        voc_path = os.path.join(vocab_dir, 'voc_' + off + '.txt')
        vocabs = open(voc_path, 'r').readlines()

    # print('image:', impath)
    img = cv2.imread(impath)
    cnt += 1
    tic = time.time()
    predictions, bounding_boxes = coco_demo.run_on_opencv_image(img)
    toc = time.time()

    print('time cost:', str(toc - tic)[:6], '|', str(cnt) + '/' + str(num_images), off)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes_np = bounding_boxes.bbox.data.cpu().numpy()
    # cfg.MODEL.RRPN.GT_BOX_MARGIN / cfg.MODEL.ROI_REC_HEAD.BOXES_MARGIN
    bboxes_np[:, 2:4] /= cfg.MODEL.RRPN.GT_BOX_MARGIN

    width, height = bounding_boxes.size

    box_scores = bounding_boxes.get_field("scores").data.cpu().numpy()

    # print('has prob:', bounding_boxes.has_field('word_probs'))
    word_labels = []
    if bounding_boxes.has_field('word_probs'):

        word_probs_np = bounding_boxes.get_field('word_probs').data.cpu().numpy()
        #print("wp:", word_probs_np.shape)
        #word_probs_np = np.squeeze(word_probs_np, axis=1)
        # print('word_probs_np', word_probs_np.shape)
        labels_np = np.argmax(word_probs_np, axis=-1)
        probs_np = np.max(word_probs_np, axis=-1)

        ws_thres = 0.8
        bs_thres = 0.7

        w_scores = []
        for i in range(probs_np.shape[0]):
            label_seq = labels_np[i]
            prob_seq = probs_np[i]
            w_score = np.mean(prob_seq[label_seq > 0])
            w_scores.append(w_score)
        w_scores = np.array(w_scores)
        # print("w_scores:", w_scores, w_scores.shape)

        # filter by w_scores
        if w_scores.shape[0] > 0:

            label_np_low_ws = labels_np[w_scores <= ws_thres]
            bboxes_np_low_ws = bboxes_np[w_scores <= ws_thres]
            box_scores_lw = box_scores[w_scores <= ws_thres]

            labels_np_high_ws = labels_np[w_scores > ws_thres]
            bboxes_np_high_ws = bboxes_np[w_scores > ws_thres]

            label_np_lwhb = label_np_low_ws[box_scores_lw > bs_thres]
            bboxes_np_lwhb = bboxes_np_low_ws[box_scores_lw > bs_thres]
            if cfg.TEST.MODE == "DET":
                labels_np = np.concatenate([labels_np_high_ws, label_np_lwhb], axis=0)
                bboxes_np = np.concatenate([bboxes_np_high_ws, bboxes_np_lwhb], axis=0)
            elif cfg.TEST.MODE == "E2E":
                labels_np = labels_np_high_ws
                bboxes_np = bboxes_np_high_ws

        for i in range(labels_np.shape[0]):
            l0 = alphabet[labels_np[i, 0] - 1]
            label_strs = l0 if labels_np[i, 0] > 0 else ''
            # print('labels_np:', np.max(labels_np[i]))
            for j in range(1, labels_np.shape[1]):
                # print('labels_np[i, j]:', labels_np[i, j])
                l = alphabet[labels_np[i, j] - 1]
                l_last = alphabet[labels_np[i, j-1] - 1]
                if labels_np[i, j] > 0 and l_last != l :
                    label_strs += l
                # pass

            word_labels.append(label_strs)
    if vis:
        pil_image = vis_image_with_words(Image.fromarray(img), bboxes_np, None, None, None) #word_labels vocabs
        pil_image.save(os.path.join(vis_dir, image))
        # time.sleep(20)
    if cfg.TEST.MODE == "E2E":
        write_result_ICDAR_RRPN2polys_with_words(image[:-4], bboxes_np, word_labels, vocabs, result_dir, height, width)
    elif cfg.TEST.MODE == "DET":
        write_result_ICDAR_RRPN2polys(image[:-4], bboxes_np, threshold=0.7, result_dir=result_dir, height=height,
                                  width=width)
    # write_result_ICDAR_RRPN2polys(image[:-4], bboxes_np, threshold=0.7, result_dir=result_dir, height=height, width=width)
    #im_file, dets, threshold, result_dir, height, width
    #cv2.imshow('win', predictions)
    #cv2.waitKey(0)

if dataset_name == 'IC15':
    zipfilename = os.path.join(result_dir, 'submit_' + str(iter) + '.zip')
    if os.path.isfile(zipfilename):
        print('Zip file exists, removing it...')
        os.remove(zipfilename)
    zip_dir(result_dir, zipfilename)
    comm = 'curl -i -F "submissionFile=@' + zipfilename + '" http://127.0.0.1:8080/evaluate'
    # print(comm)
    print(os.popen(comm, 'r'))
