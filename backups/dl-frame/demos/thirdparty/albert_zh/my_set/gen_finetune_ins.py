# -*- coding: gb18030 -*-
 
"""
FileName: my_set/gen_finetune_ins.py
Date: 2020-01-05 16:47:38
"""
import random
file_pre = "pretrain.txt"
#file_pre = "pretrain.txt.demo"
file_train = "train.txt"
file_eval = "dev.txt"
file_test = "test.txt"
max_ins_per_user = 30

instances = []
with open(file_train, 'wb') as fout_train, \
         open(file_eval, "wb") as fout_eval, \
         open(file_pre, "rb") as fin_pre, \
         open(file_test, "wb") as fout_test:
    uid = 1
    dic = {}
    for line in fin_pre:
        line = line.strip("\n")
        if line == "":
            uid += 1
        else:
            dic.setdefault(str(uid), set())
            for att in line.split(";"):
                dic[str(uid)].add(att)
    keys = dic.keys()
    idx = 0
    for u in keys:
        readlist = list(dic[u])
        for repeat_idx in xrange(0, max_ins_per_user):
            ## rand 5 atts
            cur_set = set()
            for i in xrange(0, 5):
                rand_pos = random.randint(0, len(readlist) - 1)
                cur_set.add(readlist[rand_pos])
            for pred in dic[u]:
                if pred not in cur_set:
                    break
            pred_str = readlist[random.randint(0, len(readlist) - 1)]
            cur_str = ";".join(list(cur_set))
            pos_ins = "\t".join([cur_str, pred_str, "1"])

            # rand a neg user
            rand_neg = random.randint(0, len(keys) - 1)
            neg_readlist = list(dic[keys[rand_neg]])
            neg_str = neg_readlist[random.randint(0, len(neg_readlist) - 1)]
            neg_ins = "\t".join([cur_str, neg_str, "0"])

            instances.append(pos_ins)
            instances.append(neg_ins)
        idx += 1
    total_cnt = len(instances)
    train_cnt = int(total_cnt * 0.8)
    eval_cnt = int(total_cnt * 0.1)
    test_cnt = int(total_cnt * 0.1)
    for q in xrange(0, train_cnt):
        fout_train.write(instances[q] + "\n")
    for q in xrange(train_cnt, train_cnt + eval_cnt):
        fout_eval.write(instances[q] + "\n")
    for q in xrange(train_cnt + eval_cnt, train_cnt + eval_cnt + test_cnt):
        fout_test.write(instances[q] + "\n")
