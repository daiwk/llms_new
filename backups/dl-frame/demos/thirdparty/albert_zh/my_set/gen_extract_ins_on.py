#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
 
"""
FileName: gen_extract_ins_on.py
Date: 2020-03-07 23:10:18
"""

file = "./feed_data/tag_news_with_video_vector"
#file = "./feed_data/tag_news_with_video_vector.head"

fin = open(file, 'r')
file_ins = "./test_extract.txt"
with open(file_ins, 'w') as fout_ins:
  fout_ins.write("xxx\txxx\t1\n")
  for line in fin:
    line = line.strip("\n")
    vals = line.split(" ")
    tag = vals[0]
    try:
      tag.decode("utf8")
    except:
      continue
    fout_ins.write(tag + "\t\t1\n")
