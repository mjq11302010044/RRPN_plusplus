import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import cv2
import numpy as np
from maskrcnn_benchmark.config import cfg
from demo.predictor import ICDARDemo, RRPNDemo
from maskrcnn_benchmark.utils.visualize import vis_image, write_result_ICDAR_RRPN2polys, zip_dir, merge_closest
from PIL import Image
import time
import json
from tqdm import tqdm
# from Pascal_VOC import eval_func
from demo.link_boxes import merge
from demo.hmean import compute_hmean
import time
from tensorboardX import SummaryWriter
import argparse


def res2json(result_dir):
    res_list = os.listdir(result_dir)

    res_dict = {}

    for rf in tqdm(res_list):
        if rf[-4:] == '.txt':
            respath = os.path.join(result_dir, rf)
            reslines = open(respath, 'r').readlines()
            reskey = rf[4:-4]
            res_dict[reskey] = [{'points':np.array(l.replace('\n', '').split(','), np.int).reshape(-1, 2).tolist()} for l in reslines]

    json_tarf = os.path.join(result_dir, 'res.json')

    if os.path.isfile(json_tarf):
        print('Json file found, removing it...')
        os.remove(json_tarf)

    j_f = open(json_tarf, 'w')
    json.dump(res_dict, j_f)
    print('json dump done', json_tarf)

    return json_tarf

def padding32(img):

    o_height, o_width = img.shape[:2]

    padding_im = np.zeros((o_height + 32, o_width + 32, 3))
    padding_im[16:16 + o_height, 16:16 + o_width] = img

    return padding_im.astype(np.uint8)


def gotest(setting_weights, config_file, remove_output_dir=False):

    # config_file = "configs/arpn/e2e_rrpn_R_50_C4_1x_train_AFPN_RT.yaml" #'configs/rrpn/e2e_rrpn_R_50_C4_1x_LSVT_test_RFPN.yaml' #'#"configs/ICDAR2019_det_RRPN/e2e_rrpn_R_50_C4_1x_LSVT_val_4scales_angle_norm.yaml" #e2e_rrpn_R_50_C4_1x_ICDAR13_15_trial_test.yaml

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    if remove_output_dir:
        cfg.merge_from_list(["OUTPUT_DIR", ""])
    opts = ["MODEL.WEIGHT", setting_weights]
    cfg.merge_from_list(opts)
    # cfg.MODEL.WEIGHT = setting_weights
    # cfg.freeze()

    vis = True
    merge_box = cfg.TEST.MERGE_BOX
    result_dir = os.path.join('results', config_file.split('/')[-1].split('.')[0], cfg.MODEL.WEIGHT.split('/')[-1].split('.')[0])

    if merge_box:
        result_dir += '_merge_box'

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    else:
        print(os.popen("rm -rf " + result_dir).read())
        os.makedirs(result_dir)
    coco_demo = RRPNDemo(
        cfg,
        min_image_size=1260,
        confidence_threshold=0.4,
    )
    #
    dataset_name = cfg.TEST.DATASET_NAME

    print("dataset_name:", dataset_name)

    testing_dataset = {
        'LSVT': {
            'testing_image_dir': '../datasets/LSVT/train_full_images_0/train_full_images_0/',
            'off': [0, 3000]
        },
        'ArT': {
            'testing_image_dir': '../datasets/ArT/ArT_detect_train/train_images',
            'off': [4000, 5603]
        },
        'ReCTs':{
            'testing_image_dir': '',
            'off': [0, 1000000]
        },
        'MLT':{
            'testing_image_dir': '/mnt/nvme/Dataset/MLT/val/image/',
            'off': [0, 1000000]
        },
        'MLT_test':{
            'testing_image_dir':
                ['/mnt/nvme/Dataset/MLT/test/MLT19_TestImagesPart1/TestImagesPart1',
                 '/mnt/nvme/Dataset/MLT/test/MLT19_TestImagesPart2/TestImagesPart2'],
            'off': [0, 1000000]
        },
        'IC15':{
            'testing_image_dir':
                '../dataset/ICDAR15/ch4_test_images/',
            'off': [0, 500],
            'gt_dir':"../dataset/ICDAR15/Challenge4_Test_Task4_GT"
        },
        'docVQA':{
            'testing_image_dir':
                '../dataset/docVQA/docvqa_train_v0.1/documents/',
            'off': [4000, 4330],
            # 'gt_dir':"../dataset/ICDAR15/Challenge4_Test_Task4_GT"
        },
        'docVQA_TEST': {
            'testing_image_dir':
                '../dataset/docVQA/test/documents/',
            'off': [0, 1000000]  # [4000, 4330],
            # 'gt_dir':"../dataset/ICDAR15/Challenge4_Test_Task4_GT"
        },
        'docVQA_VAL':{
            'testing_image_dir':
                '../dataset/docVQA/val/documents/',
            'off': [0, 1000000]
        }
    }

    image_dir = testing_dataset[dataset_name]['testing_image_dir']
    # vocab_dir = testing_dataset[dataset_name]['test_vocal_dir']
    off_group = testing_dataset[dataset_name]['off']
    # load image and then run prediction
    # image_dir = '../datasets/ICDAR13/Challenge2_Test_Task12_Images/'
    # imlist = os.listdir(image_dir)[off_group[0]:off_group[1]]

    image_vis_dir = "vis_results_" + dataset_name + "/"
    if not os.path.isdir(image_vis_dir):
        os.makedirs(image_vis_dir)

    print('************* META INFO ***************')
    print('config_file:', config_file)
    print('result_dir:', result_dir)
    print('image_dir:', image_dir)
    print('weights:', cfg.MODEL.WEIGHT)
    print('merge_box:', merge_box)
    print('***************************************')

    imlist = []

    if type(image_dir) == list:
        for dir in image_dir:
            tmp_list = os.listdir(dir)
            for im_name in tmp_list:
                imlist.append(os.path.join(dir, im_name))
    else:
        imlist = os.listdir(image_dir)

    imlist.sort()

    num_images = len(imlist)
    cnt = 0

    if dataset_name in ['ReCTs', 'MLT']:
        for image in imlist:
            # image = 'gt_' + str(idx) + '.jpg'
            if type(image_dir) == list:
                impath = image
            else:
                impath = os.path.join(image_dir, image)

            # print('image:', impath)
            img = cv2.imread(impath)
            cnt += 1
            tic = time.time()
            predictions, bounding_boxes = coco_demo.run_on_opencv_image(img)
            toc = time.time()

            print('time cost:', str(toc - tic)[:6], '|', str(cnt) + '/' + str(num_images))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bboxes_np = bounding_boxes.bbox.data.cpu().numpy()
            # bboxes_np[:, 2:4] /= cfg.MODEL.RRPN.GT_BOX_MARGIN

            if merge_box:
                bboxes_np_reverse = bboxes_np.copy()
                bboxes_np_reverse[:, 2:4] = bboxes_np_reverse[:, 3:1:-1]
                bboxes_np_reverse = merge(bboxes_np_reverse)
                bboxes_np_reverse[:, 2:4] = bboxes_np_reverse[:, 3:1:-1]
                bboxes_np = bboxes_np_reverse

            width, height = bounding_boxes.size

            if vis:
                pil_image = vis_image(Image.fromarray(img), bboxes_np)
                # pil_image.show()

                # time.sleep(20)
            else:

                results_prefix = image[:-4].replace('tr_', '') if dataset_name == 'MLT' else image[:-4]

                write_result_ICDAR_RRPN2polys(results_prefix, bboxes_np, threshold=0.7, result_dir=result_dir, height=height,
                                              width=width)
            # im_file, dets, threshold, result_dir, height, width
            # cv2.imshow('win', predictions)
            # cv2.waitKey(0)
    else:
        num_images = min(off_group[1], len(imlist)) - off_group[0]
        for idx in range(off_group[0], min(off_group[1], len(imlist))):

            if dataset_name == "IC15":
                image = 'img_' + str(idx+1) + '.jpg' #idx
            else:
                image = imlist[idx]
            impath = os.path.join(image_dir, image)
            # print('image:', impath)
            img = cv2.imread(impath)
            # img = padding32(img)

            cnt += 1
            tic = time.time()
            predictions, bounding_boxes = coco_demo.run_on_opencv_image(img)
            toc = time.time()

            print('time cost:', str(toc - tic)[:6], '|', str(cnt) + '/' + str(num_images), image)

            bboxes_np = bounding_boxes.bbox.data.cpu().numpy()
            # bboxes_np[:, 2:4] /= cfg.MODEL.RRPN.GT_BOX_MARGIN

            # bboxes_np[:, :2] = bboxes_np[:, :2] - 16.

            if merge_box:
                bboxes_np_reverse = bboxes_np.copy()
                bboxes_np_reverse[:, 2:4] = bboxes_np_reverse[:, 3:1:-1]
                bboxes_np_reverse = merge(bboxes_np_reverse)
                bboxes_np_reverse[:, 2:4] = bboxes_np_reverse[:, 3:1:-1]
                bboxes_np = bboxes_np_reverse

            width, height = bounding_boxes.size

            scores = bounding_boxes.get_field("scores").data.cpu().numpy()

            if bboxes_np.shape[0] > 0:
                # merge_keep = merge_closest(bboxes_np, scores, 0.81)
                # bboxes_np = bboxes_np[merge_keep]
                # scores = scores[merge_keep]
                pass
            if vis:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img)
                # print("bboxes_np:", bboxes_np)

                if "gt_dir" in testing_dataset[dataset_name]:
                    gt_file = os.path.join(testing_dataset[dataset_name]["gt_dir"], "gt_" + image.split(".")[0] + ".txt")

                    gt_lines = open(gt_file, "r").readlines()
                    gt_boxes = []
                    for line in gt_lines:
                        splts = line.replace("\r", "").replace("\n", "").split(",")
                        if len(splts) > 8:
                            gt_boxes.append(np.array(splts[:8]).astype(np.int32))

                        pil_image = vis_image(pil_image, np.array(gt_boxes), mode=1, repre="poly")

                    # gt_image.save(os.path.join(image_vis_dir, "gt_" + image))
                pil_image = vis_image(pil_image, bboxes_np, mode=2)

                pil_image.save(os.path.join(image_vis_dir, image))
                # cv2.imwrite(os.path.join(image_vis_dir, "mask_" + image), predictions)
                # pil_image.show()
                # time.sleep(20)
            # else:

            # print("bboxes_np", bboxes_np.shape)

            write_result_ICDAR_RRPN2polys(image[:-4], bboxes_np, threshold=0.7, result_dir=result_dir, height=height, width=width)
            #im_file, dets, threshold, result_dir, height, width
            # print("predictions:", predictions.shape)
            # cv2.imshow('win', predictions)
            # cv2.waitKey(0)

    del coco_demo

    if dataset_name == 'IC15':

        zipfilename = os.path.join(result_dir, 'submit_' + config_file.split('/')[-1].split('.')[0] + '_' + cfg.MODEL.WEIGHT.split('/')[-1].split('.')[0] + '.zip')
        if os.path.isfile(zipfilename):
            print('Zip file exists, removing it...')
            os.remove(zipfilename)
        zip_dir(result_dir, zipfilename)

        comm = 'curl -i -F "submissionFile=@' + zipfilename + '" http://127.0.0.1:8080/evaluate'
        print(comm)
        print(os.popen(comm, 'r'))

        res_dict = compute_hmean(zipfilename)
        del res_dict["per_sample"]
        print(res_dict)

        return res_dict

    elif dataset_name == 'MLT':
        zipfilename = os.path.join(result_dir, 'submit_' + config_file.split('/')[-1].split('.')[0] + '_' + cfg.MODEL.WEIGHT.split('/')[-1].split('.')[0] + '.zip')
        if os.path.isfile(zipfilename):
            print('Zip file exists, removing it...')
            os.remove(zipfilename)
        zip_dir(result_dir, zipfilename)
        comm = 'curl -i -F "submissionFile=@' + zipfilename + '" http://127.0.0.1:8080/evaluate'
        # print(comm)
        print(os.popen(comm, 'r'))

    elif dataset_name == 'LSVT':
        # input_json_path = 'results/e2e_rrpn_R_50_C4_1x_LSVT_val/model_0190000/res.json'
        gt_json_path = '../datasets/LSVT/train_full_labels.json'
        # to json
        input_json_path = res2json(result_dir)
        eval_func(input_json_path, gt_json_path)
    elif dataset_name == 'ArT':
        # input_json_path = 'results/e2e_rrpn_R_50_C4_1x_LSVT_val/model_0190000/res.json'
        gt_json_path = '../datasets/ArT/ArT_detect_train/train_labels.json'
        # to json
        input_json_path = res2json(result_dir)
        eval_func(input_json_path, gt_json_path)
    else:
        pass

    return None


def update_check(config_file):

    cfg.merge_from_file(config_file)
    model_dir = cfg.OUTPUT_DIR
    out_split = model_dir.split("/")
    tensorboard_dir = os.path.join("tensorboard", out_split[-1] if len(out_split[-1]) > 0 else out_split[-2])
    print("tensorboard_dir:", tensorboard_dir)
    if not os.path.isdir(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    result_writer = SummaryWriter(tensorboard_dir)

    last_model_list = os.listdir(model_dir)
    last_ckpt = ""
    while True:
        new_list = os.listdir(model_dir)
        ckpt_file = os.path.join(model_dir, "last_checkpoint")
        if not os.path.isfile(ckpt_file):
            continue
        new_last_ckpt = open(ckpt_file
            ,
            "r"
        ).readlines()[0].replace("\n", "")
        # if len(new_list) > len(last_model_list):

        if last_ckpt != new_last_ckpt:
            # continue

            last_iter = int(new_last_ckpt.split("/")[-1].split(".")[0].split("_")[-1])

            print("New ckpt found, testing...", last_ckpt)
            res_dict = gotest(last_ckpt, config_file)

            if not res_dict is None:
                # for key in res_dict["method"]:
                #     result_writer.add_scalar('results/' + key, res_dict["method"][key] * 100, global_step=last_iter)
                result_writer.add_text('Results',
                                '''hmean: {} \n precision:{} \n recall: {}\n'''.format(
                                    res_dict["method"]["hmean"],
                                    res_dict["method"]["precision"],
                                    res_dict["method"]["recall"]),
                                last_iter)
            print("Testing done")

            # last_model_list = new_list
            last_ckpt = new_last_ckpt
        time.sleep(10)

if __name__ == "__main__":

    # cfg.merge_from_file("configs/arpn/e2e_rrpn_R_50_C4_1x_test_AFPN.yaml")
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    args = parser.parse_args()
    assert os.path.isfile(args.config_file), "Configure file doesn't exist..."



    gotest(
        "",
        args.config_file,
        True
    )
    # tensorboard_dir = "./tensorboard"
    # writer = SummaryWriter(tensorboard_dir)
    # print("tensorboard_dir:", tensorboard_dir)
    # for i in range(100):
    #     writer.add_text('markdown Text', '''hmean: {} \n precision:{} \n recall: {}\n'''.format(i, 1, 2), 100)
    # update_check(args.config_file)