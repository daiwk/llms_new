from tensorflow.contrib import predictor
import numpy as np
export_dir = "./test_model_2/1590926782"

IteratorV2 = [0] * 5
IteratorV2[0] = (np.zeros((64, 128))).astype('int32')
IteratorV2[1] = (np.zeros((64, 128))).astype('int32')
IteratorV2[2] = (np.zeros((64, 128))).astype('int32')



with open('./input_ids', 'r') as fin_input, \
    open('./input_mask', 'r') as fin_mask, \
    open('./segment_ids', 'r') as fin_token:
    i = 0
    for line in fin_input:
        line = line.strip("\n").split(" ")
        #print(line, 'qq')
        for j in range(len(line)):
            #print(i,j)
            IteratorV2[0][i][j] = int(line[j])
        i += 1

    i = 0
    for line in fin_mask:
        line = line.strip("\n").split(" ")
        #print(line, 'qq')
        for j in range(len(line)):
            #print(i,j)
            IteratorV2[1][i][j] = int(line[j])
        i += 1

    i = 0
    for line in fin_token:
        line = line.strip("\n").split(" ")
        for j in range(len(line)):
            #print(i,j, 'xx')
            IteratorV2[2][i][j] = int(line[j])
        i += 1



predict_fn = predictor.from_saved_model(export_dir)
predictions = predict_fn(
            {"input_ids": IteratorV2[0], "input_mask": IteratorV2[1], "segment_ids": IteratorV2[2]})
print(predictions['probabilities'])
