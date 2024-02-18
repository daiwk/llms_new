import paddle.fluid as fluid
import numpy as np
import math

exe = fluid.Executor(fluid.CPUPlace())
path = "./pd-model/inference_model"

[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)

print(feed_target_names, fetch_targets)

IteratorV2 = [0] * 5
##IteratorV2[0] = fluid.layers.data(dtype='int32', shape=[64, 128], name='IteratorV2_0', append_batch_size=False) ## input_ids 先查word_embeddings，得到dim=128，再查word_embeddings_2，得到dim=312
##IteratorV2[1] = fluid.layers.data(dtype='int32', shape=[64, 128], name='IteratorV2_1', append_batch_size=False)
##IteratorV2[2] = fluid.layers.data(dtype='int32', shape=[64], name='IteratorV2_2', append_batch_size=False)
##IteratorV2[3] = fluid.layers.data(dtype='int32', shape=[64], name='IteratorV2_3', append_batch_size=False)
##IteratorV2[4] = fluid.layers.data(dtype='int32', shape=[64, 128], name='IteratorV2_4', append_batch_size=False) ## token_type_embeddings

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

results = exe.run(inference_program,
        feed={feed_target_names[0]: IteratorV2[0], feed_target_names[1]: IteratorV2[1], feed_target_names[2]: IteratorV2[2]},
        fetch_list=fetch_targets)
#print(results[0])
for x in results[0]:
    print(x[0], x[1])
#    print(softmax(x[0], x[1]), softmax(x[1], x[0]))
