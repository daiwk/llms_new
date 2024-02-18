#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
########################################################################
 
"""
FileName: parse_res.py
Date: 2019-12-25 02:01:42
"""

file_pred = "../albert_my_set_checkpoints/test_results.tsv"
##file_pred = "../albert_my_set_checkpoints_fake/test_results.tsv"
file_mid = "./mid_res"
file_out = "./out.res"
g_max_save = 300

with open(file_pred, "r") as fin_pred, \
         open(file_mid, "r") as fin_mid, \
         open(file_out, "w") as fout:
    idx = 1
    dic_pred = {}
    for line in fin_pred:
        line = line.strip("\n").split('\t')
        score = float(line[1])
        dic_pred[str(idx)] = score
        idx += 1
    dic_mid = {}
    for line in fin_mid:
        line = line.strip("\n").split('\t')
        dic_mid[line[0]] = [line[1], line[3]]



    dic_out = {}
    for idx in dic_pred:
        score = dic_pred[idx]
        uid, att = dic_mid[idx]
        dic_out.setdefault(uid, [])
        dic_out[uid].append((att, score))

       
    for uid in dic_out:
        uid_res = dic_out[uid]
        sorted_res = sorted(uid_res, key=lambda x:x[1], reverse=True)
        iix = 0
        for itm in sorted_res:
            if iix > g_max_save:
                break
            fout.write("\t".join([str(k) for k in [uid, itm[0], itm[1]]])+"\n")
            iix += 1
