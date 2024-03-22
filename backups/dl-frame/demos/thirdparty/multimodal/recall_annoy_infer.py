#encoding=utf8
"""recall_annoy"""
import sys
from annoy import AnnoyIndex
import numpy as np
import os
vec_dim = int(os.environ["DIM"])
width = int(os.environ["WIDTH"])
height = int(os.environ["HEIGHT"])
model_type = os.environ["MODEL_TYPE"]

annoy_file = "img.annoy." + model_type 
ann_model = AnnoyIndex(vec_dim, metric='angular')
ann_model.load(annoy_file)

idx2nid_file = "img.idx." + model_type
vec_file = "img_infer.vec." + model_type

top_k = 100
g_upper_threshold = 0.01
g_lower_threshold = 0.01
g_max_print_cnt = 200
g_max_total_print_cnt = 10

dic_idx2nid = {}
with open(idx2nid_file, 'rb') as fin_idx2nid:
    for line in fin_idx2nid:
        line = line.strip("\n").split("\t")
        idx = int(line[0])
        nid = line[1]
        dic_idx2nid[idx] = nid

#req_nid_info = "badcase.txt"
#req_dic_nidinfo = {}
#with open(req_nid_info, 'rb') as fin_nidinfo_req:
#    for line in fin_nidinfo_req:
#        line = line.strip("\n").split("\t")
##        if len(line) != 6:
##            continue
#        nid = line[0]
#        info = "xx1"
#        info += "</td><td>" + "xx2"
#        info += "</td><td>" + line[1][:100] # title
#        info += "</td><td>" + "<img src='%s'/>" % (line[4])
#        req_dic_nidinfo[nid] = info

nid_info = "info.res"
dic_nidinfo = {}
with open(nid_info, 'rb') as fin_nidinfo:
    for line in fin_nidinfo:
        line = line.strip("\n").split("\t")
        if len(line) != 6:
            continue
        nid = line[0]
        info = line[1]
        info += "</td><td>" + line[2]
        info += "</td><td>" + line[3][:100] # title
        #info += "</td><td>" + "<img src='%s' height=146 width=218/>" % (line[4])
        info += "</td><td>" + "<img src='%s' height=%d width=%d/>" % (line[4], height, width)
        dic_nidinfo[nid] = info

nid_info = "bad_case.res"
req_dic_nidinfo = {}
with open(nid_info, 'rb') as fin_nidinfo_req:
    for line in fin_nidinfo_req:
        line = line.strip("\n").split("\t")
        if len(line) != 6:
            continue
        nid = line[0]
        info = line[1]
        info += "</td><td>" + line[2]
        info += "</td><td>" + line[3][:100] # title
        #info += "</td><td>" + "<img src='%s'/>" % (line[4])
        info += "</td><td>" + "<img src='%s' height=%d width=%d/>" % (line[4], height, width)
        req_dic_nidinfo[nid] = info


with open(vec_file, 'rb') as fin:

    print '<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />'
    print '<table style="BORDER-COLLAPSE: collapse" borderColor=#000000' \
                       ' cellSpacing=0 align=center text-align=left bgColor=#ffffff border=1 >'
 
    print '<th>req_nid</th>'\
        '<th>req_type</th>'\
        '<th>req_cate</th>'\
        '<th>req_title</th>'\
        '<th>req_img</th>'\
        '<th>res_nid</th>'\
        '<th>res_type</th>'\
        '<th>res_cate</th>'\
        '<th>res_title</th>'\
        '<th>res_img</th>'\
        '<th>score</th>'

    total_idx = 0
    for line in fin:
        line = line.strip("\n").split("\t")
        nid = line[0]
        vec = np.array(map(float, line[1].split(" ")))
        #print nid
        res_idxs, res_distances = ann_model.\
                    get_nns_by_vector(vec, top_k, include_distances=True)
        #print res_idxs, 'mmm'
#        if res_distances[0] > g_upper_threshold:
#            continue
        i = 0
        for idx in res_idxs:
            #print idx, 'qqq'
            if i >= g_max_print_cnt:
                break
            res_nid = dic_idx2nid[idx]
            if res_nid == nid:
                i += 1
                continue
            dist = res_distances[i]
#            if dist > g_lower_threshold:
#                break
            try:
                #print req_dic_nidinfo.keys()
                xtype = req_dic_nidinfo[nid].split("</td><td>")[0]
                res_xtype = dic_nidinfo[res_nid].split("</td><td>")[0]
                print "<tr><td>" + "</td><td>".join(str(x) for x in [xtype + "_" + nid, req_dic_nidinfo[nid], res_xtype + "_" + res_nid, dic_nidinfo[res_nid], dist]) + "</tr>"
            except Exception as e:
                i += 1
                #print e, "oooooooo"
                continue
            i += 1

        if total_idx >= g_max_print_cnt:
            break
        total_idx += 1
    
    print "</table>"
