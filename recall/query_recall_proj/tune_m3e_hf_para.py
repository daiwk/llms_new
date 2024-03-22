import sys
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

file_in = os.environ["file_in"]

def write_res(ins, out):
    return InputExample(texts=[ins, out], label=1.0)

#max_lines = 100
idx = 0
train_examples = []
with open(file_in, 'r') as fin:
    for line in fin:
#        if idx >= max_lines:
#            break
        query, product, cate1, cate2, cate3 = line.strip("\n").split("\1")
        train_examples.append(write_res(query, product))
        train_examples.append(write_res(query, cate1))
        train_examples.append(write_res(query, cate2))
        train_examples.append(write_res(query, cate3))
        idx += 1
#Define the model. Either from scratch of by loading a pre-trained model

model = SentenceTransformer(os.environ["model_name"])

#Define your train examples. You need more than just two examples...
#train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
#    InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]

#Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16, num_workers=4)
train_loss = losses.CosineSimilarityLoss(model)

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, warmup_steps=100)
model.save(os.environ["new_model_name"])

