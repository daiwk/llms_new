#!/usr/bin/env python
# -*- coding: gb18030 -*-
 
"""
FileName: my_set/gen_pretrain_data.py
Date: 2019-12-28 22:51:55
"""
import sys

line_set = set()
for line in sys.stdin:
    line = line.strip("\n").split("\t")
    if line[0] in line_set:
        continue
    if line[-1] == "1":
        for x in line[0].split(","):
            print x + ","
##        print line[1]
        print ""
        line_set.add(line[0])
