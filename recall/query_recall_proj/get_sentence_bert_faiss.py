#encoding=utf8

import sys
import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

cur_vec_file = sys.argv[1]
query_file = sys.argv[2]
model_name = sys.argv[3]
out_name = sys.argv[4]

vec_dim = int(os.environ["vec_dim"])
model = SentenceTransformer(model_name)

top_k = 100
g_per_query_res_min_cnt = int(os.environ["g_per_query_res_min_cnt"])
g_cos_threshold = float(os.environ["g_cos_threshold"])
#g_cos_threshold = 0.001
#g_max_print_cnt = 100

import re


dic_idx2name = {}

def check_validquery(x):
    reg = "^[0-9]+|[a-zA-Z]+|\-+$" 
    matches = re.match(reg, x)
    if matches is None:
        return True
    else:
        return False

def build_index():
    #index = faiss.IndexFlatL2(vec_dim)  # build the index
    index = faiss.index_factory(vec_dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    if os.environ["FLAG"] == "on":
        index = faiss.index_cpu_to_all_gpus(index)

    idx = 0
    vecs = []
    with open(cur_vec_file, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip("\n").split("\t")
            name = line[0]
            author_id = line[2]
            dic_idx2name[idx] = name + "\2" + author_id
            vec = np.array([float(x) for x in line[-1].split(" ")]).astype('float32')
            vecs.append(vec)
            idx += 1
    np_vecs = np.array(vecs)
    #print(np_vecs.shape)
    faiss.normalize_L2(np_vecs)
    #print(np_vecs.shape)
    index.add(np_vecs)
    return index

def get_res(x_lst_names, pool, fout):
    results = model.encode_multi_process(x_lst_names, pool)
    xidx = 0
    faiss.normalize_L2(results)
    res_dist_mat, res_idx_mat = index.search(results, top_k)
    
    for res_idxs in res_idx_mat:
        res_distances = res_dist_mat[xidx]
        query = x_lst_names[xidx]
        if not check_validquery(query):
            xidx += 1
            continue
        
        req_info = [query]
        i = 0
        res_infos = []
        for idx in res_idxs:
            #if i >= g_max_print_cnt:
            #     break
            res_name = dic_idx2name[idx]
            cos = res_distances[i]
            if cos < g_cos_threshold:
                break
            if "旗舰店" in query and cos < 0.8:
                break
            res_info = [res_name]
            res_info.append(str(cos))
            res_infos.append("\2".join(res_info))
            i += 1
    
        if len(res_infos) < g_per_query_res_min_cnt:
            xidx += 1
            continue
        out_list = req_info + ["\1".join(res_infos)]
        #print "\t".join(x.encode("utf8") for x in out_list)
        fout.write("\t".join(x for x in out_list) + "\n")            
        xidx += 1
        


if __name__ == "__main__":
    index = build_index()

    pool = model.start_multi_process_pool()
    
    batch_size = int(os.environ["query_batch_size"])
    with open(query_file, 'r', encoding="utf8") as fin_query, \
        open(out_name, 'w') as fout:
        x_lst_names = []
        
        for line in fin_query:
            line = line.strip("\n").split("\t")
            query = line[0]
            #print(query)
            x_lst_names.append(query)
            if len(x_lst_names) > batch_size:
                get_res(x_lst_names, pool, fout)
                x_lst_names = []
        if len(x_lst_names) > 0:
            get_res(x_lst_names, pool, fout)   
    
    model.stop_multi_process_pool(pool)
            


