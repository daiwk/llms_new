#!/usr/bin/env python
# encoding: utf-8
"""
build_annoy
"""
import json
import pickle
import numpy as np
from annoy import AnnoyIndex
import os
model_type = os.environ["MODEL_TYPE"]
annoy_file = "img.annoy." + model_type 
idx2nid_file = "img.idx." + model_type
vec_file = "img.vec." + model_type
fout_idx2nid = open(idx2nid_file, "wb")
#vec_dim = 12288
vec_dim = int(os.environ["DIM"])
annoy_lib = AnnoyIndex(vec_dim, metric='angular')
def build_annoy():
    """
    build_annoy
    """
    idx = 0
    with open(vec_file, 'rb') as fin:
        for line in fin:
            line = line.strip("\n").split("\t")
            nid = line[0]
            if len(line[1].split(" ")) != vec_dim:
                continue
            vec = np.array(map(float, line[1].split(" ")))
            fout_idx2nid.write('%d\t%s\n' % (idx, nid))
            annoy_lib.add_item(idx, vec.tolist())
            idx += 1
        annoy_lib.build(10)
        annoy_lib.save(annoy_file)
if __name__ == "__main__":
    build_annoy()
