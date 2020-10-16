# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict
import logging

import torch

from maskrcnn_benchmark.utils.imports import import_file


def align_and_update_state_dicts(model_state_dict, loaded_state_dict):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """

    logger = logging.getLogger(__name__)

    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    size_cnt = 0
    '''
    for key in loaded_keys:
        size = loaded_state_dict[key].size()
        size_acc = 1
        for s in size:
            size_acc *= s
        size_acc *= 4
        # print()

        logger.info(
            "loaded key:" + " " + key + " " + str(size) + " " + str(size_acc / float(1048576))
        )

        size_cnt += size_acc / float(1048576)

    size_loaded = size_cnt
    size_cnt = 0

    for key in current_keys:
        size = model_state_dict[key].size()
        size_acc = 1
        for s in size:
            size_acc *= s
        size_acc *= 4
        # print("current key:", key, size_acc / float(1048576))

        logger.info(
            "current key:" + " " + key + " " + " " + str(size) + " " + str(size_acc / float(1048576))
        )

        size_cnt += size_acc / float(1048576)
    
    logger.info(
        "All size:" + " " + str(size_cnt) + "MB / " + str(size_loaded) + "MB"
    )
    '''
    # print("All size:", size_cnt, "MB", "/", size_loaded ,"MB")
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"

    updated = []

    for idx_new, idx_old in enumerate(idxs.tolist()):

        # if "rpn.head.abox_tower.0.conv.weights" in current_keys[idx_new]loaded_keys[idx_old]:

        if idx_old == -1:
            logger.info(
                "We don't have key:" + current_keys[idx_new] + " discard it..."
            )
            print(logger.info(
                "We don't have key:" + current_keys[idx_new] + " discard it..."
            ))
            
            continue

        updated.append(current_keys[idx_new])

        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]

        tar_model_size = model_state_dict[key].size()
        src_model_size = loaded_state_dict[key_old].size()

        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            ) + "\t" + str(tar_model_size) + " & " + str(src_model_size)
        )

        print(log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            ) + "\t" + str(tar_model_size) + " & " + str(src_model_size))

        if tar_model_size != src_model_size:
            logger.info(
                "key: " + key + " size not matched, discard..."
            )
            print("key: " + key + " size not matched, discard...")
            continue
        model_state_dict[key] = loaded_state_dict[key_old]

    for key in model_state_dict:
        if not key in updated:
            print("Key", key, "not loaded...")

    '''
    special_match = {
        "rpn.head.abox_tower.0.conv.weight": "rpn.head.conv.weight",
        "rpn.head.abox_tower.0.conv.bias": "rpn.head.conv.bias"
    }

    for new_key in special_match:
        old_key = special_match[new_key]
        if not new_key in model_state_dict:
            continue
        if not old_key in loaded_state_dict:
            continue

        model_state_dict[new_key] = loaded_state_dict[old_key]

        print("For special match: {} / {}".format(new_key, old_key))

        logger.info(
            "For special match: {} / {}".format(new_key, old_key)
        )
    '''

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict)

    # use strict loading
    model.load_state_dict(model_state_dict)
