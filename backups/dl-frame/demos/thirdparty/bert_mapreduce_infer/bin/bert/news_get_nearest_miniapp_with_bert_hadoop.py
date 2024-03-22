#!/usr/bin/env python
# encoding: utf-8
"""
hadoop given a news, search bert miniapp annoy for nearest minapps
"""

import json
import sys
sys.path.append("./bin/")
import utils
import globals
from annoy import AnnoyIndex
import numpy as np


dic_idx2nid = utils.load_bert_idx2nid_dict_from_miniapp_annoy()

vec_dim = int(sys.argv[1])

ann_model = AnnoyIndex(vec_dim, metric='angular')
ann_model.load(globals.bert_miniapp_annoy_file)


top_k = 10

stop_words = set()
with open(globals.stopwords_file, 'r') as f:
    for line in f:
        stop_words.add(line.strip())

def news_get_nearest_miniapp_with_bert():
    """
    news_get_nearest_miniapp_with_bert
    """
    xidx = 0

    chosen_token = "[CLS]"
    use_layer = -1
    for line in sys.stdin:
        line = line.strip("\n")
        js = json.loads(line)
        linex_index = js["linex_index"]
        nid = linex_index
        for fea in js["features"]:
            if fea["token"] == chosen_token:
                for layer in fea["layers"]:
                    if layer["index"] == use_layer:
                        vec = layer["values"]
                        nid = linex_index

        need_lst = ["title", "url", "news_category", "news_sub_category", "news_attention"]

        res_idxs, res_distances = ann_model.\
                get_nns_by_vector(vec, top_k, include_distances=True) # will find the top_k nearest neighbors

        if res_distances[0] > globals.g_upper_threshold:
            continue

        req_info = [nid]
        i = 0
        for idx in res_idxs:
            if i >= globals.g_max_print_cnt:
                break
            res_nid = dic_idx2nid[idx]
            dist = res_distances[i]
            if dist > globals.g_lower_threshold:
                break
            res_info = [res_nid]
            res_info.append(str(dist))
            out_list = req_info + res_info
            print "\t".join(x.encode("utf8") for x in out_list)
            i += 1
        xidx += 1


def process():
    """
    process
    """
    news_get_nearest_miniapp_with_bert()


if __name__ == "__main__":

    process()

