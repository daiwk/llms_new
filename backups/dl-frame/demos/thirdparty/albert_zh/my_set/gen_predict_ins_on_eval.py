#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
########################################################################
 
"""
FileName: deal.py
Date: 2019-12-25 01:41:45
"""
from common import get_profile_new

flag_demo = True
flag_demo = False
if flag_demo == True:
    flag = ".demo"
else:
    flag = ""
file_x = "./user_att.txt" + flag
file_x = "./for_qcal.text.eval.utf8"
file_mid_res = "mid_res" 
file_ins = "./test.txt"
max_ins_len = 128 
max_b_len = 20
max_a_len = max_ins_len - max_b_len
max_a_len = 4*3
max_a_len = 15*3 # base
max_a_len = 25*3
#max_a_len = 30*3
ins = ""
dic = {}
white_att_set = set()
black_att_set = set()

with open("./black_att", 'rb') as fin_black:
    for line in fin_black:
        black_att_set.add(line.strip("\n"))

#with open("../white.txt.sorted", 'r') as fin_white:
##with open("../white.processed.txt", 'r') as fin_white:
#with open("../news_sub_cates", 'r') as fin_white:
with open("../mid_names", 'r') as fin_white:
    for line in fin_white:
        line = line.strip("\n").split('\t')
        att = line[0]
        white_att_set.add(att)
with open(file_x, 'r') as fin, \
    open(file_mid_res, 'w') as fout_mid, \
    open(file_ins, 'w') as fout_ins:
    
    for line in fin:
        line = line.strip("\n").split(' ')
        for xx in line:
            if xx.startswith("{user_id}:{"):
                uid = xx.split("{user_id}:{")[-1].split("}")[0]
##            if xx.startswith("{user_news_attention_top5}:{"):
##                att = xx.split("{user_news_attention_top5}:{")[-1].split("}")[0]
            if xx.startswith("{user_meditation_news_attention_top1_1}:{"):
                att = xx.split("{user_meditation_news_attention_top1_1}:{")[-1].split("}")[0]
            elif xx.startswith("{user_meditation_news_attention_top2_3}:{"):
                att = xx.split("{user_meditation_news_attention_top2_3}:{")[-1].split("}")[0]
            elif xx.startswith("{user_meditation_news_attention_top4_10}:{"):
                att = xx.split("{user_meditation_news_attention_top4_10}:{")[-1].split("}")[0]
            elif xx.startswith("{user_meditation_news_cat1_top1_1}:{"):
                att = xx.split("{user_meditation_news_cat1_top1_1}:{")[-1].split("}")[0]
            elif xx.startswith("{user_meditation_news_cat1_top2_3}:{"):
                att = xx.split("{user_meditation_news_cat1_top2_3}:{")[-1].split("}")[0]
            elif xx.startswith("{user_meditation_news_cat2_top1_1}:{"):
                att = xx.split("{user_meditation_news_cat2_top1_1}:{")[-1].split("}")[0]
            elif xx.startswith("{user_meditation_news_cat2_top2_3}:{"):
                att = xx.split("{user_meditation_news_cat2_top2_3}:{")[-1].split("}")[0]
##            elif xx.startswith("{user_meditation_news_cat2_top4_10}:{"):
##                att = xx.split("{user_meditation_news_cat2_top4_10}:{")[-1].split("}")[0]

####            elif xx.startswith("{user_meditation_video_cat1_top1_1}:{"):
####                att = xx.split("{user_meditation_video_cat1_top1_1}:{")[-1].split("}")[0]
####            elif xx.startswith("{user_meditation_video_cat1_top2_3}:{"):
####                att = xx.split("{user_meditation_video_cat1_top2_3}:{")[-1].split("}")[0]
##            elif xx.startswith("{user_meditation_video_cat1_top4_10}:{"):
##                att = xx.split("{user_meditation_video_cat1_top4_10}:{")[-1].split("}")[0]

##            elif xx.startswith("{user_meditation_video_attention_top1_1}:{"):
##                att = xx.split("{user_meditation_video_attention_top1_1}:{")[-1].split("}")[0]
##            elif xx.startswith("{user_meditation_video_attention_top2_3}:{"):
##                att = xx.split("{user_meditation_video_attention_top2_3}:{")[-1].split("}")[0]
##            elif xx.startswith("{user_meditation_video_attention_top4_10}:{"):
##                att = xx.split("{user_meditation_video_attention_top4_10}:{")[-1].split("}")[0]
            else:
                continue
            if att in black_att_set:
                continue
            att = att.split("-->>")[0]
            dic.setdefault(uid, {})
            dic[uid].setdefault("u_represent", "")
            dic[uid].setdefault("atts", [])
            dic[uid].setdefault("attset", set())
            dic[uid].setdefault("used_attset", set())
            
            if att in dic[uid]["attset"]:
                continue
           
            if len(dic[uid]["u_represent"] + ";" + att) < max_a_len:
                if dic[uid]["u_represent"] != "": #age + "," + gender + ",":
                    dic[uid]["u_represent"] += ";" + att
                else:
                    dic[uid]["u_represent"] += att
                dic[uid]["used_attset"].add(att)

            dic[uid]["atts"].append(att)
            dic[uid]["attset"].add(att)

    idx = 1
    fout_ins.write("xxx\txxx\t1\n")
    for uid in dic:
        cand_attset = dic[uid]["attset"] - dic[uid]["used_attset"]
        ###for att in cand_attset: # use user att
        for att in white_att_set: # use white att 
            mid_res = "\t".join([str(idx), uid, dic[uid]["u_represent"], att]) + '\n'
            ins_res = "\t".join([dic[uid]["u_represent"], att, "1"]) + '\n'
            fout_mid.write(mid_res)
            fout_ins.write(ins_res)
            idx += 1
    

