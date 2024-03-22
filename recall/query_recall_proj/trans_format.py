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
    with open(recall_file, "r") as fin, \
        open(recall_file + ".save", "w") as fout:
        for line in fin:
            line = line.strip("\n").split("\t")
            query = line[0]
            author_lst = line[1].split("\1")
            mapping = {}
            x_lst = []
            for pair in author_lst:
                author, author_id, score = pair.split("\2")
                x_str = ":".join([author_id, "%.6f"% float(score)])
                x_lst.append(x_str)
                fout.write("\t".join([query, author, author_id, score]) + "\n")


if __name__ == '__main__':
    trans_format()

