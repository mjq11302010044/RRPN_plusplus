# RRPN++: Guidance Towards More Accurate Scene Text Detection
Report can be viewed at: https://arxiv.org/abs/2009.13118

# Hightlights

- 89.5 F-measure in single scale in ICDAR 2015 benchmark (https://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=1&m=78081).
- 92.0 F-measure in single scale in ICDAR 2013 and testing speed can reach 13.3 fps with 640px (tested in single GPU of GTX 1080Ti).
- Adopting RRoI Align in Detectron2, Reimplement RoI-Rotate in FOTS (see ROIAlignRotatedKeep).
- Text Spotter with Transformer (training and testing).

## Environments
- Ubuntu 16.04
- Cuda 10 or 9
- python >=3.5
- pytorch >= 1.7
- Other packages like numpy, cv2.


![alt text](demo/visualization.png "Results from IC15 testing set")

## Highlights
- **From original repo:** In pytorch 1.0, Somehow faster than original repo in both training and inference.
- **Training and evaluation checked:** Testing in IC15 with training data in {IC13, IC15, IC17mlt}, and receives Fscore of 83% vs. 81% in caffe repo.
- **What's new:** RRoI Pooling is replaced with RRoI Alignment(bilinear interpolation for sampling), FPN structure supported, easy to change various backbones for different purposes.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Configuring your dataset
- Your dataset path can be set in `$RRPN_ROOT/maskrcnn_benchmark/config/paths_catalog.py`. We implemented interface for {IC13, IC15, IC17mlt, LSVT, ArT} for common use(Start from line 96):
```bash
...
 "RRPN_train": {  # including IC13 and IC15
            'dataset_list':{
                # 'IC13': 'Your dataset path',
                ...
            },
            "split": 'train'
        },
...
```
- Add your dataset in detector?
You need to form a dict array as follows:
```bash
im_info = {
    'gt_classes': your class_id array,
    'max_classes': your class_id array,
    'image': path to access one image,
    'boxes': rotate box in {cx, cy, w, h, θ},
    'flipped': Not supported, just False, 
    'gt_overlaps': overlaps fill with 1 (gt with gt),
    'seg_areas': H * W for an rbox,
    'height': height of an image,
    'width': width of an image,
    'max_overlaps': overlaps fill with 1 (gt with gt),
    'rotated': just True
}
```
Examples can be seen in `$RRPN_ROOT/maskrcnn_benchmark/data/rotation_series.py`
Your data API should be add to the variable `DATASET`:
```bash
DATASET = {
    'IC13':get_ICDAR2013,
    'IC15':get_ICDAR2015_RRC_PICK_TRAIN,
    'IC17mlt':get_ICDAR2017_mlt,
    ...
    'Your Dataset Name': 'Your Dataset API'
}
```

- Add your dataset in spotter?
You need to form a dict array as follows:
```bash
im_info = {
    'gt_classes': your class_id array,
    'max_classes': your class_id array,
    'image': path to access one image,
    'boxes': rotate box in {cx, cy, w, h, θ},
    'flipped': Not supported, just False, 
    'gt_overlaps': overlaps fill with 1 (gt with gt),
    'seg_areas': H * W for an rbox,
    'height': height of an image,
    'width': width of an image,
    'gt_words': words of each box,
    'max_overlaps': overlaps fill with 1 (gt with gt),
    'rotated': just True
}
```
Examples can be seen in `$RRPN_ROOT/maskrcnn_benchmark/data/rrpn_e2e_series.py`
Your data API should be add to the variable `DATASET`:
```bash
DATASET = {
    'IC13':get_ICDAR2013,
    'IC15':get_ICDAR2015_RRC_PICK_TRAIN,
    'IC17mlt':get_ICDAR2017_mlt,
    ...
    'Your Dataset Name': 'Your Dataset API'
}
```

## Training 
```bash
# create your data cache directory
mkdir data_cache
```

Train a detector of RRPN++

```bash
# In your root of RRPN
python tools/train_net.py --config-file=configs/arpn/e2e_rrpn_R_50_C4_1x_train_AFPN_RT_LERB.yaml
```

Train a spotter (Used in RRPN++ report and we strongly recommand to use) of RRPN++

```bash
# In your root of RRPN
python tools/train_net.py --config-file=configs/arpn_E2E/e2e_rrpn_R_50_C4_1x_train_AFPN_RT_LERB_Spotter.yaml
```

- Multi-GPU phase is not testing yet, be careful to use GPU more than 1.

## Testing
- Using  `$RRPN_ROOT/demo/ICDAR19_eval_script.py` or `$RRPN_ROOT/demo/rrpn_e2e_infer.py`(Strongly recommanded) to test images you want. The demo will generate a text for your detected coodinates.
- Showing the detected image by ture the variable `vis` to True.

- By adding the following setting into your configure yaml to test the datasets, or you can re-implement the file to test your images.
- One of the configure file we recommand is `$RRPN_ROOT/configs/arpn_E2E/e2e_rrpn_R_50_C4_1x_test_AFPN_RT_LERB_Spotter.yaml`
- Choose the dataset you want to evaluate on.

```bash
TEST:
  DATASET_NAME: "IC15" # Choice can be "IC15", "LSVT" and so on
  MODE: "DET" # DET for detection evaluation or E2E for recognition results in the spotter
```

## Pretrained Weights for Training and Testing
- Here we provide some pretained weights for testing in google drive:

```bash
  ICDAR 2015 F-measure 89.0 : https://drive.google.com/file/d/1nv-ZjbYBj8ePZRa_fAhbHvzm7HqSxPWK/view?usp=sharing
  SynthText pretrained: https://drive.google.com/file/d/1RuqEg8pkN_l-4_KhOHzOeZcFiihQlxUT/view?usp=sharing
```


## More Results 

## Final 
- Enjoy it with all the codes.
- Citing us if you find it work in your projects.

```
@article{ma2020rrpn++,
  title={RRPN++: Guidance Towards More Accurate Scene Text Detection},
  author={Ma, Jianqi},
  journal={arXiv preprint arXiv:2009.13118},
  year={2020}
}
```

## Special Thanks
- My family for FIRM SUPPORT of devices and power supply.
- Jingye Chen's (https://github.com/JinGyeSetBirdsFree) support of the code and report.
