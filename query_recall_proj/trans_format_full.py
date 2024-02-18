#encoding=utf8
import os
import sys
import math
import time
from collections import defaultdict
from multiprocessing import Process
import time

def trans_format():
    recall_file = sys.argv[1]
    query_file = sys.argv[2]
    with open(recall_file, "r") as fin, \
        open(query_file, "r") as fin_q, \
        open(recall_file + ".save", "w") as fout:
        dic_res = {}
        for line in fin:
            line = line.strip("\n").split("\t")
            query = line[0]
            author_lst = line[1].split("\1")
            mapping = {}
            x_lst = []
            for pair in author_lst:
                author, author_id, score = pair.split("\2")
                #x_str = ":".join([author_id, "%.6f"% float(score)])
                x_str = "\t".join([author, score])
                x_lst.append(x_str)
            dic_res[query] = x_lst
        for line in fin_q:
            query = line.strip("\n")
            max_cnt = 10
            print(query)
            for idx in range(0, max_cnt):
                if query in dic_res and idx < len(dic_res[query]):
                    fout.write("\t".join([query, dic_res[query][idx]]) + "\n")
                else:
                    fout.write("\t".join([query, "null", "null"]) + "\n")

if __name__ == '__main__':
    trans_format()

