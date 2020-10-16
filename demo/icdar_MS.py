import os
import numpy as np
from rotation.rotate_polygon_nms import rotate_gpu_nms
from maskrcnn_benchmark.utils.visualize import write_result_ICDAR_RRPN2polys, zip_dir, merge_closest
import cv2
import imageio
import time
from maskrcnn_benchmark.config import cfg
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from maskrcnn_benchmark.config import cfg
from demo.predictor import ICDARDemo, RRPNDemo
from hmean import compute_hmean
import pickle
# rotate_gpu_nms(np.array(np.hstack((ch_proposals, score_np[..., np.newaxis])), np.float32), nms_thresh, GPU_ID)  # D

# [640, 960, 1280]

ms_res_dirs = {
    "result_prefix": 'results/e2e_rrpn_R_50_C4_1x_LSVT_val_AFPN_Higher_IOU/model_0225000',
    "scales": [896],#, 768, 896, 1024
    'ori_im_dir':'../dataset/ICDAR15/ch4_test_images/',
}

'''
ms_res_dirs = {
    'S': 'results/e2e_rrpn_X_101_C4_1x_MLT_val_RFPN/R-50_S',
    'M': 'results/e2e_rrpn_X_101_C4_1x_MLT_val_RFPN/R-50_M',
    'L': 'results/e2e_rrpn_X_101_C4_1x_MLT_val_RFPN/R-50_L',
    'ori_im_dir':'/mnt/nvme/Dataset/MLT/val/image/',
}
'''
fuse_resdir = './results/e2e_rrpn_R_50_C4_1x_MLT_val_AFPN/fuse_R_50_MS_test' # fuse_X_101_MS_test
fuse_resdir_4save = './results/e2e_rrpn_R_50_C4_1x_MLT_val_AFPN/fuse_R_50_MS_test_4_save'
if not os.path.isdir(fuse_resdir):
    print('Build result dirs...')
    os.makedirs(fuse_resdir)
    print('done')

if not os.path.isdir(fuse_resdir_4save):
    print('Build save dirs...')
    os.makedirs(fuse_resdir_4save)
    print('done')


def get_rboxes(path):
    con_list = open(path, 'r').readlines()
    rlist = [p2r(np.array(ele.replace('\n', '').split(',')).astype(np.float32)) for ele in con_list]

    # print("rlist:", path, len(rlist))
    # if len(con_list) < 1:
    #     return np.array(rlist)
    return np.array(rlist).reshape(len(con_list), 7)


def p2r(poly):
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
    return np.array([x_ctr, y_ctr, width, height, angle] + poly[8:].tolist())


def r2p(rbox):

    cx, cy, w, h, angle = rbox[0:5]
    lt = [cx - w / 2, cy - h / 2, 1]
    rt = [cx + w / 2, cy - h / 2, 1]
    lb = [cx - w / 2, cy + h / 2, 1]
    rb = [cx + w / 2, cy + h / 2, 1]

    pts = []

    # angle = angle * 0.45

    pts.append(lt)
    pts.append(rt)
    pts.append(rb)
    pts.append(lb)

    angle = -angle

    # if angle != 0:
    cos_cita = np.cos(np.pi / 180 * angle)
    sin_cita = np.sin(np.pi / 180 * angle)

    M0 = np.array([[1, 0, 0], [0, 1, 0], [-cx, -cy, 1]])
    M1 = np.array([[cos_cita, sin_cita, 0], [-sin_cita, cos_cita, 0], [0, 0, 1]])
    M2 = np.array([[1, 0, 0], [0, 1, 0], [cx, cy, 1]])
    rotation_matrix = M0.dot(M1).dot(M2)

    rotated_pts = np.dot(np.array(pts), rotation_matrix)

    return np.concatenate([rotated_pts[:, :2].reshape(-1), rbox[-1:]], axis=0)


def rnms(rboxes, nms_thresh, GPU_ID=0):
    ch_proposals = rboxes.copy()
    ch_proposals[:, 2:4] = ch_proposals[:, 3:1:-1]
    # x,y,h,w,a

    # print('ch_proposals:',ch_proposals.shape)
    # print('score_np:', score_np.shape)

    if ch_proposals.shape[0] < 1:
        return rboxes

    keep = rotate_gpu_nms(ch_proposals.astype(np.float32), nms_thresh, GPU_ID)  # D

    return keep


def ms_eval(ms_res_dirs, cfg):

    merge_box = cfg.TEST.MERGE_BOX
    '''
    for scale in ms_res_dirs["scales"]:

        print("*************** Scale:", scale, " ***************")

        result_dir = ms_res_dirs["result_prefix"] + "_" + str(scale)

        model = RRPNDemo(
        cfg,
        min_image_size=scale,
        confidence_threshold=0.6,
        )

        image_list = os.listdir(ms_res_dirs['ori_im_dir'])
        num_images = len(image_list)
        cnt = 0
        for idx in range(len(image_list)):
            image = 'img_' + str(idx+1) + '.jpg'
            impath = os.path.join(ms_res_dirs['ori_im_dir'], image)
            # print('image:', impath)
            img = cv2.imread(impath)
            cnt += 1
            tic = time.time()
            predictions, bounding_boxes = model.run_on_opencv_image(img)
            toc = time.time()

            print('time cost:', str(toc - tic)[:6], '|', str(cnt) + '/' + str(num_images))

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bboxes_np = bounding_boxes.bbox.data.cpu().numpy()
            scores = bounding_boxes.get_field("scores").data.cpu().numpy()
            # ms_scores = bounding_boxes.get_field("mask_score").data.cpu().numpy()
            # bboxes_np[:, 2:4] /= cfg.MODEL.RRPN.GT_BOX_MARGIN

            # if merge_box:
            #    bboxes_np_reverse = bboxes_np.copy()
            #    bboxes_np_reverse[:, 2:4] = bboxes_np_reverse[:, 3:1:-1]
            #   bboxes_np_reverse = merge(bboxes_np_reverse)
            #   bboxes_np_reverse[:, 2:4] = bboxes_np_reverse[:, 3:1:-1]
            #    bboxes_np = bboxes_np_reverse

            width, height = bounding_boxes.size

            # if vis:
            #     pil_image = vis_image(Image.fromarray(img), bboxes_np)
            #     pil_image.save(os.path.join(image_vis_dir, image))
                # pil_image.show()
                # time.sleep(20)
            # else:
            write_result_ICDAR_RRPN2polys(image[:-4],
                                          bboxes_np,
                                          threshold=0.7,
                                          result_dir=result_dir,
                                          height=height,
                                          width=width,
                                          cls_prob=scores,
                                          # ms_scores=ms_scores,
                                          scale=[scale for i in range(bboxes_np.shape[0])])

        del model
    '''
    f_list = os.listdir(ms_res_dirs["result_prefix"] + "_" + str(ms_res_dirs["scales"][0]))

    nms_thres = 0.1

    cnt = 0

    if len(os.listdir(fuse_resdir)) > 0:
        print(os.popen("rm -rf " + fuse_resdir).read())
        os.makedirs(fuse_resdir)

    for resname in f_list:
        if resname[-4:] == '.txt':
            # print('resname:', resname)
            all_boxes = []
            for scale in ms_res_dirs["scales"]:
                result_dir = ms_res_dirs["result_prefix"] + "_" + str(scale)
                all_boxes.append(get_rboxes(os.path.join(result_dir, resname)))
            # sboxes = get_rboxes(os.path.join(ms_res_dirs['S'], resname)).reshape(-1, 6)
            # mboxes = get_rboxes(os.path.join(ms_res_dirs['M'], resname)).reshape(-1, 6)
            # lboxes = get_rboxes(os.path.join(ms_res_dirs['L'], resname)).reshape(-1, 6)
            # print('shape:', sboxes.shape, mboxes.shape, lboxes.shape)

            # idx = int(((int(resname[-9:-4]) - 1) / 5000)) + 1
            imdir = os.path.join(ms_res_dirs['ori_im_dir'], resname[4:-4])
            # imdir = os.path.join(ms_res_dirs['ori_im_dir']) + '/tr_' + resname[4:-4]

            # print('resname:', resname[-9:-4])
            im = cv2.imread(imdir + '.jpg')
            '''
            
            if im is None:
                im = cv2.imread(imdir + '.JPG')
            if im is None:
                im = cv2.imread(imdir + '.png')
            if im is None:
                tmp = imageio.mimread(imdir + '.gif')
                im_tmp = np.array(tmp)
                im_tmp = im_tmp[0]
                im = im_tmp[:, :, 0:3]
            '''
            height, width = im.shape[:2]

            all_boxes = np.concatenate(all_boxes, axis=0)
            # all_boxes[:, -1] = 1 - all_boxes[:, -1]

            # cls_score, ms_score, scales

            arg = np.argsort(-all_boxes[:, 6])
            all_boxes = all_boxes[arg]
            keep = rnms(np.concatenate([all_boxes[:, :5], all_boxes[:, 5:6]], axis=1), nms_thres)
            # print(keep)
            # all_boxes = all_boxes[keep]

            # merge_keep = merge_closest(all_boxes[:, :5], all_boxes[:, 5], 0.85)
            # all_boxes = all_boxes[merge_keep]

            all_boxes = all_boxes[all_boxes[:, 5] > 0.8]
            # all_boxes = all_boxes[all_boxes[:, 6] > 0.55]
            # print('all_boxes:', all_boxes.shape)
            cnt += 1
            print('cnt:', str(cnt) + '/' + str(len(f_list)), resname)

            write_result_ICDAR_RRPN2polys(resname[4:-4], all_boxes, 0.7, fuse_resdir_4save, height, width,
                                          cls_prob=all_boxes[:, 5],
                                          # ms_scores=all_boxes[:, 6],
                                          scale=all_boxes[:, 6]
                                          )
            write_result_ICDAR_RRPN2polys(resname[4:-4], all_boxes, 0.7, fuse_resdir, height, width,
                                          # cls_prob=all_boxes[:, 5],
                                          # ms_scores=all_boxes[:, 6],
                                          # scale=all_boxes[:, 7]
                                          )

    zipfilename = os.path.join(fuse_resdir, 'submit_MS_test.zip')
    if os.path.isfile(zipfilename):
        print('Zip file exists, removing it...')
        os.remove(zipfilename)
    zip_dir(fuse_resdir, zipfilename)
    # res_dict = compute_hmean(zipfilename)

    # res_samples = res_dict["per_sample"]
    # print("res_sample:", res_samples.keys())

    pkl_cache = "res_cache.pkl"

    # if not os.path.isfile(pkl_cache):
    res_dict = compute_hmean(zipfilename)
    # res_samples = res_dict["per_sample"]
    res_samples = res_dict["per_sample"]

    pkl_f = open(pkl_cache, "wb")
    pickle.dump(res_samples, pkl_f)
    pkl_f.close()
    # else:
    #    pkl_f = open(pkl_cache, "rb")
    #    res_samples = pickle.load(pkl_f)

    # print("res_sample:", res_samples.keys())

    '''
    comm = 'curl -i -F "submissionFile=@' + zipfilename + '" http://127.0.0.1:8080/evaluate'
    # print(comm)
    print(os.popen(comm, 'r'))
    '''

    vis_eval(res_samples, ms_res_dirs['ori_im_dir'], fuse_resdir_4save)

COLORS = [
    (255, 255, 255), # False Det
    (0, 255, 204), # True Det
    (0, 255, 102), # GT
    (255, 102, 102) # Low IoU
]


def vis_eval(res_samples, imdir, res_file_dir):
    # image_list = os.listdir(imdir)

    match_log_file = "match_log.txt"

    match_str = ""

    for res_key in res_samples.keys():
        print("imfile:", res_key)

        match_str += "match_ID: " + res_key + "\n"

        res_sample = res_samples[str(res_key)]

        gts = np.array(res_sample["gtPolPoints"])
        dets = np.array(res_sample["detPolPoints"])

        iou_matrix = np.array(res_sample["iouMat"]).astype(np.float32)
        # print("iou_matrix:", iou_matrix.shape, len(gts), len(dets))
        # max IoU det per gt
        gt_det_argmax = np.argmax(iou_matrix, axis=1)

        impath = os.path.join(imdir, "img_" + res_key + ".jpg")
        img = cv2.imread(impath)

        respath = os.path.join(res_file_dir, "res_img_" + res_key + ".txt")

        reslines = open(respath, "r").readlines()

        # print("compare: res", reslines[0:5])
        # print("compare: det", dets[0:5])

        for i in range(iou_matrix.shape[1]):
            if not i in gt_det_argmax.tolist():
                cls, scale = reslines[i].replace("\n", "").replace("\r", "").split(",")[8:]
                str_show = "{:.2f} / {}".format(float(cls), int(float(scale)))
                # cv2.putText(img, str_show, (int(dets[i][0]), int(dets[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[0], 1)
                cv2.polylines(img, [dets[i].astype(np.int32).reshape((-1, 1, 2))], True, color=COLORS[0], thickness=1)

        # if len(dets) < 0:
        #     continue

        for i in range(iou_matrix.shape[0]):
            gt_pts = np.array(gts[i])
            # print("det", i, gt_det_argmax.shape, dets.shape)
            # cls, scale = reslines[gt_det_argmax[i]].replace("\n", "").replace("\r", "").split(",")[8:]
            # str_show = "{:.2f} / {:.2f} / {:d}".format(float(cls), float(ms), int(float(scale)))
            # cv2.putText(img, str_show, (int(dets[gt_det_argmax[i]][0]), int(dets[gt_det_argmax[i]][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[1], 1)
            det = np.array(dets[gt_det_argmax[i]])
            cv2.polylines(img, [gt_pts.astype(np.int32).reshape((-1, 1, 2))], True, color=COLORS[2], thickness=1)
            cv2.polylines(img, [det.astype(np.int32).reshape((-1, 1, 2))],
                          True,
                          color=COLORS[1] if iou_matrix[i, gt_det_argmax[i]] > 0.5 else COLORS[3],
                          thickness=2)

            det_match = np.where(iou_matrix[i] > 0)
            for mi in det_match[0]:
                # matched = False
                # print("mi:", det_match)
                line = reslines[mi].replace("\n", "").replace("\r", "")

                rbox = p2r(np.array(line.split(",")).astype(np.float32))

                match_str += line + "," + \
                             str(iou_matrix[i, mi]) + "," + \
                             str(mi in gt_det_argmax.tolist() and iou_matrix[i, mi] > 0.5) + "," + \
                             str(int(rbox[2] * rbox[3])) + "," + \
                             "\n"
            match_str += "\n"
        match_str += "\n"

        # cv2.imshow("win", img)
        # cv2.waitKey(0)
        image_vis_dir = "vis_results/"
        cv2.imwrite(os.path.join(image_vis_dir, "img_" + res_key + ".jpg"), img)

    log_f = open(match_log_file, "w")
    log_f.write(match_str)
    log_f.close()

if __name__ == "__main__":
    config_file = 'configs/arpn/e2e_rrpn_R_50_C4_1x_test_AFPN.yaml'  # '#"configs/ICDAR2019_det_RRPN/e2e_rrpn_R_50_C4_1x_LSVT_val_4scales_angle_norm.yaml" #e2e_rrpn_R_50_C4_1x_ICDAR13_15_trial_test.yaml

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    ms_eval(ms_res_dirs, cfg)