#encoding=utf8
import os
import sys
import math
import time
from collections import defaultdict
from multiprocessing import Process
import time


__redis_client = None

def get_redis_client():

    global __redis_client
    if __redis_client == None:
        __redis_client = "xx" 

    return __redis_client

def write_redis():
    redis_client = get_redis_client()
    recall_file = sys.argv[1]
    prefix = sys.argv[2]
    with open(recall_file, "r") as fin, \
        open(recall_file + ".save." + prefix, "w") as fout:
        for line in fin:
            line = line.strip("\n").split("\t")
            query = line[0]
            author_lst = line[1].split("\1")
            mapping = {}
            author_id_lst = []
            x_lst = []
            for pair in author_lst:
                author, author_id, score = pair.split("\2")
                x_str = ":".join([author_id, "%.6f"% float(score)])
                x_lst.append(x_str)
                fout.write("\t".join([query, author, author_id, score]) + "\n")
                author_id_lst.append(author_id)
            
            redis_key = prefix + query
            table_name = "[ecom_gip_query2cate]"
            x_lst_str = ";".join(x_lst)
            redis_client.set(table_name + redis_key, x_lst_str)
            #redis_client.expire(redis_key, 30 * 86400)
            if len(author_id_lst) > 0:
                x = redis_client.get(table_name + redis_key)
                if x is not None:
                    print redis_key, x[: 100]
            if os.environ["DEBUG"] == "debug":
                break
#            for author_id in author_id_lst:
#                x = redis_client.zscore(redis_key, author_id)
#                print redis_key, author_id, x



if __name__ == '__main__':
    write_redis()

