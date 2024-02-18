import sys
import os
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

file_in = sys.argv[1]

def get_cate_list(cate_file):
    cate_lst = []
    with open(cate_file, 'r') as fin:
        for line in fin:
            cate_lst.append(line.strip("\n"))
    return cate_lst

cate1_lst = get_cate_list("./dev_dir/first.txt")
cate2_lst = get_cate_list("./dev_dir/second.txt")
cate3_lst = get_cate_list("./dev_dir/third.txt")


def write_res(ins, out, label):
    return InputExample(texts=[ins, out], label=label)

def neg_sample(train_examples, in_str, cate_lst, cur_cate, neg_num):
    idx = 0
    while idx < neg_num:
        rand_idx = random.randint(0, len(cate_lst) - 1)
        neg_cate = cate_lst[rand_idx]
        if cate_lst[idx] != cur_cate:
            train_examples.append(write_res(in_str, neg_cate, 0.))
            print(in_str, neg_cate, rand_idx, cur_cate)
        idx += 1

def process_ins(train_examples, query, product, cate1, cate2, cate3):
    #train_examples.append(write_res(query, product))
    train_examples.append(write_res(product, cate1, 1))
    train_examples.append(write_res(product, cate2, 1))
    train_examples.append(write_res(product, cate3, 1))
    train_examples.append(write_res(query, cate1, 1))
    train_examples.append(write_res(query, cate2, 1))
    train_examples.append(write_res(query, cate3, 1))
    neg_sample(train_examples, product, cate1_lst, cate1, neg_num=3)
    neg_sample(train_examples, product, cate2_lst, cate2, neg_num=3)
    neg_sample(train_examples, product, cate3_lst, cate3, neg_num=3)
    neg_sample(train_examples, query, cate1_lst, cate1, neg_num=3)
    neg_sample(train_examples, query, cate2_lst, cate2, neg_num=3)
    neg_sample(train_examples, query, cate3_lst, cate3, neg_num=3)


if __name__ == "__main__":

    
    #max_lines = 100
    idx = 0
    train_examples = []
    with open(file_in, 'r') as fin:
        for line in fin:
            #if idx >= max_lines:
            #    break
            query, product, cate1, cate2, cate3 = line.strip("\n").split("\1")
            process_ins(train_examples, query, product, cate1, cate2, cate3)
            idx += 1
    #Define the model. Either from scratch of by loading a pre-trained model
    
    model = SentenceTransformer(os.environ["model_name"])
    
    #Define your train examples. You need more than just two examples...
    #train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
    #    InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
    
    #Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)
    
    #Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    model.save(os.environ["new_model_name"])
    
