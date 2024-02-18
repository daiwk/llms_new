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
file_mid_res = "mid_res.train" 
file_train_ins = "./train.txt"
file_test_ins = "./test.txt"
file_eval_ins = "./eval.txt"
train_cnt = 800
test_cnt = train_cnt + 10
eval_cnt = test_cnt + 10
max_ins_len = 128 
max_b_len = 20
max_a_len = max_ins_len - max_b_len
ins = ""
dic = {}
g_max_atts = 10
with open(file_x, 'r') as fin, \
    open(file_train_ins, 'w') as fout_train_ins, \
    open(file_test_ins, 'w') as fout_test_ins, \
    open(file_eval_ins, 'w') as fout_eval_ins:
    
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
        cnt = 0 
        if len(dic[uid]["u_represent"] + "," + att) < max_a_len and cnt < g_max_atts:
            if dic[uid]["u_represent"] != "":
                dic[uid]["u_represent"] += "," + att
            else:
                dic[uid]["u_represent"] += att
            cnt += 1

        dic[uid]["atts"].append(att)
        dic[uid]["attset"].add(att)

    idx = 1
    fout_test_ins.write("xxx\txxx\t1\n")
#    fout_train_ins.write("xxx\txxx\t1\n")
#    fout_eval_ins.write("xxx\txxx\t1\n")
    for uid in dic:
        for att in dic[uid]["atts"]:
            ins_res = "\t".join([dic[uid]["u_represent"], att, "1"]) + '\n'
            if idx < train_cnt:
                fout_train_ins.write(ins_res)
            elif idx < test_cnt:
                fout_test_ins.write(ins_res)
            elif idx < eval_cnt:
                fout_eval_ins.write(ins_res)
            idx += 1
    

