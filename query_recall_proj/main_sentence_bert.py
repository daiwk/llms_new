#encoding=utf8
import os
import tensorflow as tf
import sys
import numpy as np
import cityhash

from sentence_transformers import SentenceTransformer
import sys

model_name = sys.argv[1]
file_in = sys.argv[2]

model = SentenceTransformer(model_name)

def req(x_lst, fout, pool):
    x_lst_names = []
    x_lst_info = []
    for line in x_lst:
        line = line.split("\t")
        name = line[0]
        h = cityhash.CityHash64(name)
        #real_fid = (h & ((1 << 54) - 1)) | (2 << 54)
        real_fid = h % 123333442131

        if "author" in file_in:
            author_id = line[1]
        else:
            author_id = "-1"
        x_lst_names.append(name)
        x_lst_info.append("\t".join([name, str(real_fid), author_id]))

    x_size = len(x_lst)
    results = model.encode_multi_process(x_lst_names, pool)
    idx = 0
    for res in results:
        #vec = np.mean(res, axis=0)
        out_str = x_lst_info[idx] + "\t" + " ".join("%.6f" % i for i in res)
        fout.write(out_str + "\n")
        idx += 1


x_lst = []
#batch_size = int(os.environ["author_batch_size"])
batch_size = 100
out_file = file_in + ".vec"
os.system("rm -rf %s" % out_file)

if __name__ == "__main__":
    pool = model.start_multi_process_pool()
    
    with open(file_in, 'r', encoding="utf8") as fin, \
        open(out_file, 'w') as fout:
        for line in fin:
            line = line.strip("\n")
            if len(x_lst) > batch_size:
                req(x_lst, fout, pool)
                x_lst = []
            x_lst.append(line)
        if len(x_lst) > 0:
            req(x_lst, fout, pool)
    
    model.stop_multi_process_pool(pool)

