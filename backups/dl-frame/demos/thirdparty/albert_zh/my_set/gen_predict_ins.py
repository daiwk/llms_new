#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
########################################################################
 
"""
FileName: deal.py
Date: 2019-12-25 01:41:45
"""

flag_demo = True
#flag_demo = False
if flag_demo == True:
    flag = ".demo"
else:
    flag = ""
file_x = "./user_att.txt" + flag
file_mid_res = "mid_res" 
file_ins = "./test.txt"
max_ins_len = 128 
max_b_len = 20
max_a_len = max_ins_len - max_b_len
ins = ""
dic = {}
with open(file_x, 'r') as fin, \
    open(file_mid_res, 'w') as fout_mid, \
    open(file_ins, 'w') as fout_ins:
    
    for line in fin:
        line = line.strip("\n").split('\t')
        uid = line[0]
        att = line[1]
        dic.setdefault(uid, {})
        dic[uid].setdefault("u_represent", "")
        dic[uid].setdefault("atts", [])
        dic[uid].setdefault("attset", set())
        
        if att in dic[uid]["attset"]:
            continue
        if len(dic[uid]["u_represent"] + "," + att) < max_a_len:
            if dic[uid]["u_represent"] != "":
                dic[uid]["u_represent"] += "," + att
            else:
                dic[uid]["u_represent"] += att

        dic[uid]["atts"].append(att)
        dic[uid]["attset"].add(att)

    idx = 1
    fout_ins.write("xxx\txxx\t1\n")
    for uid in dic:
        for att in dic[uid]["atts"]:
            mid_res = "\t".join([str(idx), uid, dic[uid]["u_represent"], att]) + '\n'
            ins_res = "\t".join([dic[uid]["u_represent"], att, "1"]) + '\n'
            fout_mid.write(mid_res)
            fout_ins.write(ins_res)
            idx += 1
    

