import os
import json
import rotation.rbbox_overlaps as overlaps
import numpy as np

def poly2rbox(polys):

    rbox = []

    return rbox


def merge_gt(srcjobj):
    tarjobj = {}

    for key in srcjobj:
        tarlist = []

        srcpoly = [obj[key]['box'] for obj in srcjobj[key]]

        rboxes = poly2rbox(np.array(srcpoly))