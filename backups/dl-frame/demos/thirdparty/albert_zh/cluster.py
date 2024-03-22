#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
########################################################################
 
"""
FileName: cluster.py
Date: 2020-03-08 00:21:50
"""

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans, MiniBatchKMeans

import numpy as np

cluster_num = 200
max_cnt = 1000000

def hierarchical_cluster():
    cluster = AgglomerativeClustering(n_clusters=cluster_num, affinity='euclidean', linkage='ward')
    return cluster

def kmeans():
    cluster = MiniBatchKMeans(n_clusters=cluster_num, init='k-means++', n_init=1,
                                         init_size=1000, batch_size=1000, verbose=True)
    return cluster

def run_cluster(file, file_out, delmethod):
    with open(file, 'r') as fin, \
             open(file_out, 'w') as fout:
        tag_lst = []
        X_raw = []
        idx = 0
        for line in fin:
            if idx > max_cnt:
                break
            if delmethod == "albert":
                line = line.strip("\n").split("\t")
                tag = line[0]
                tag_lst.append(tag)
                lst = [float(i) for i in line[1].split(";")]
            elif delmethod == "w2v":
                line = line.strip("\n").split(" ")
                tag = line[0]
                tag_lst.append(tag)
                lst = [float(i) for i in line[1:-1]]

            X_raw.append(lst)
            idx += 1
        X = np.array(X_raw)
        #cluster = hierarchical_cluster()
        cluster = kmeans()
        cluster.fit_predict(X)
        res_pair = []
        idx = 0
        for label in cluster.labels_:
            tag = tag_lst[idx]
            res_pair.append((tag, label))
            idx += 1
        res_pair_new = sorted(res_pair, key=lambda x:x[1])
        for x in res_pair_new:
            out_str = str(x[1]) + "\t" + x[0] + "\n"
            fout.write(out_str)

if __name__ == "__main__":
    run_cluster("features.txt", "output/albert_vec.cluster", "albert")
    run_cluster("./my_set/feed_data/tag_news_with_video_vector", "output/w2v.cluster", "w2v")
