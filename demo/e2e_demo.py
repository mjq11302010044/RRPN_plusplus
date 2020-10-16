import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from GRCNN.models import crann
from GRCNN.utils import util
from GRCNN.utils import keys
import yaml

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

from PIL import Image

_DEBUG = True

############## det model init ##############

print('Initializing detection model...')

config_file = "configs/text_maskrcnn_res50_fpn_lsvt.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.8,
)

print('done')

############################################

############## rec model init ##############

print('Initializing recognition model...')

config_yaml = './GRCNN/config/gcrnn.yml'

f = open(config_yaml)
opt = yaml.load(f)
fixed_height = 32

if opt['N_GPU'] > 1:
    opt['RNN']['multi_gpu'] = True
else:
    opt['RNN']['multi_gpu'] = False

alphabet = keys.alphabet
nClass = len(alphabet) + 1
converter = util.strLabelConverter(alphabet)
rec_model = crann.CRANN(opt, nClass).cuda()

model_path = opt['CRANN']
if os.path.isfile(opt['CRANN']):
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    start_epoch = checkpoint['epoch']
    # best_pred = checkpoint['best_pred']
    rec_model.load_state_dict(checkpoint['state_dict'])
    # print("=> loaded checkpoint '{}' (epoch {} accuracy {})"
    #       .format(model_path, checkpoint['epoch'], best_pred))

# if _DEBUG:
#    print('rec_model:', rec_model)


rec_model.eval()
print('done')

############################################


def normalize(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = Image.fromarray(img)

    img = torch.from_numpy(img).float().div(255)
    img.sub_(0.5).div_(0.5)
    return img


def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def load_img(impath):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    # response = requests.get(url)
    pil_image = Image.open(impath).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")


def rotate(image, angle, ctr):
    # convert to cv2 image
    image = np.array(image)
    (h, w) = image.shape[:2]
    scale = 1.0
    # set the rotation center
    center = (int(ctr[0]), int(ctr[1]))
    # anti-clockwise angle in the function
    M = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine(image, M, (w, h))

    return image


def improc(impath):

    # image_dir = ''
    # imlist = os.listdir(image_dir)

    # for imname in imlist:
        # impath = os.path.join(image_dir, imname)

    # from http://cocodataset.org/#explore?id=345434
    image = load_img(impath)
    # imshow(image)

    result_dict = {}

    # compute predictions
    res_im, predictions = coco_demo.run_on_opencv_image(image)

    # print('predictions:', predictions.shape)

    masks = predictions.get_field('mask')
    masks_np = masks.data.cpu().numpy()

    print('masks_np:', masks_np.shape)

    rboxes = []
    rcrops = []
    rpolys = []

    recs = []

    for i in range(masks_np.shape[0]):
        mask_np = masks_np[i][0]
        contours = cv2.findContours((mask_np * 1).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pts in contours[1]:
            poly = pts.reshape(-1, 2)
            if poly.shape[0] >= 4:
                # print('polygon:', poly.shape[0])
                rect = cv2.minAreaRect(poly)
                poly_q = np.array(cv2.boxPoints(rect), np.int)
                # print('rect:', rect)

                poly_q = poly_q.reshape(-1)
                pt1 = (int(poly_q[0]), int(poly_q[1]))
                pt2 = (int(poly_q[2]), int(poly_q[3]))
                pt3 = (int(poly_q[4]), int(poly_q[5]))
                pt4 = (int(poly_q[6]), int(poly_q[7]))

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

                rboxes.append([x_ctr, y_ctr, width, height, angle])

                rcrops.append(
                    rotate(image, -angle, (x_ctr, y_ctr))
                    [int(y_ctr-height/2):int(y_ctr+height/2), int(x_ctr-width/2):int(x_ctr+width/2)]
                )

                rpolys.append(poly.tolist())

    result_dict['polys'] = rpolys

    cnt = 0

    for crop in rcrops:
        # rec model infer

        try:
            re_img = cv2.resize(crop, (int(fixed_height / crop.shape[0] * crop.shape[1]), fixed_height))
        except Exception as e:
            print('From rec:', e)
            recs.append('')
            continue

        if _DEBUG:
            # cv2.imwrite('demo_img/crops' + str(cnt) + '.img', re_img)
            re_img_pil = Image.fromarray(cv2.cvtColor(re_img, cv2.COLOR_RGB2BGR))
            re_img_pil.save('demo_img/crops' + str(cnt) + '.jpg')
            # cv2.waitKey(0)
        cnt += 1
        # re_img_th = torch.from_numpy(np.transpose(np.expand_dims(re_img, 0), (0, 3, 1, 2))).float().cuda()
        re_img_th = normalize(re_img).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        bsz = re_img_th.size(0)

        predict = rec_model(re_img_th)
        predict_len = torch.IntTensor([predict.size(0)] * bsz)

        # if _DEBUG:
        #    print('predict:', predict.size())

        # Compute accuracy
        _, acc = predict.max(2)
        # if int(torch.__version__.split('.')[1]) < 2:
        #    acc = acc.squeeze(2)
        acc = acc.transpose(1, 0).contiguous().view(-1)
        prob, _ = F.softmax(predict, dim=2).max(2)
        probilities = torch.mean(prob, dim=1)
        sim_preds = converter.decode(acc.data, predict_len.data, raw=False)
        # if _DEBUG:
        #    print('sim_preds:', sim_preds)

        recs.append(sim_preds)

    # organize results in a dict

    result_dict['recs'] = recs

    return result_dict