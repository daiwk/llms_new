#encoding=utf8
import sys
import tensorflow as tf
import os
import ssl
import numpy as np
#from keras import backend as K

ssl._create_default_https_context = ssl._create_unverified_context
import tensorflow_hub as hub

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks
from transformers import DistilBertTokenizer, TFDistilBertModel, TFDistilBertForSequenceClassification
from transformers import AlbertTokenizer, TFAlbertModel

tf.random.set_seed(1234)

import multimodal_model


#bad_nids = set()
#with open("./badcase.txt", 'r') as finbad:
#    for line in finbad:
#        nid = line.strip("\t")[0]
#        bad_nids.add(nid)

##def read_tensor_from_image_file(file_name,
##                                input_height=299,
##                                input_width=299,
##                                input_mean=0,
##                                input_std=255):
##    """read_tensor_from_image_file"""
##    input_name = "file_reader"
##    output_name = "normalized"
##    #print file_name
##    file_reader = tf.io.read_file(file_name, input_name)
##    if file_name.endswith(".png"):
##        image_reader = tf.io.decode_png(file_reader, channels=3, name="png_reader")
##    elif file_name.endswith(".gif"):
##        image_reader = tf.squeeze(tf.io.decode_gif(file_reader, name="gif_reader"))
##    elif file_name.endswith(".bmp"):
##        image_reader = tf.io.decode_bmp(file_reader, name="bmp_reader")
##    else:
##        image_reader = tf.io.decode_jpeg(
##          file_reader, channels=3, name="jpeg_reader")
####    float_caster = tf.cast(image_reader, tf.float32)
####    dims_expander = tf.expand_dims(float_caster, 0)
####    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
####    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
##    return image_reader 

file_out = "img.vec"
input_dir = "./imgs/"

file_out = sys.argv[2]
input_dir = sys.argv[1]

white_nids = set()
with open("5000nids", 'r') as fin_white:
    for line in fin_white:
        line = line.strip("\n")
        white_nids.add(line)



dic_nidinfo = {}
if "infer" not in input_dir:
    nid_info = "info.res"
    with open(nid_info, 'r') as fin_nidinfo:
        for line in fin_nidinfo:
            line = line.strip("\n").split("\t")
            if len(line) != 6:
                continue
            nid = line[0]
            if nid not in white_nids:
                continue
            #title = line[3][:100] # title
            #title = line[2] # cate
            title = line[3] # title
            dic_nidinfo[nid] = title
else:
    nid_info = "./badcase.txt"
    with open(nid_info, 'r') as fin_nidinfo:
        for line in fin_nidinfo:
            line = line.strip("\n").split("\t")
            nid = line[0]
            #title = line[1][:100] # title
            #title = line[-1]
            title = line[1] # title
            dic_nidinfo[nid] = title

all_image_paths = []
all_nids = []
input_ids = []
attention_masks = []
g = os.walk(input_dir)
print(input_dir, g)
#g = os.walk("./imgs_test/")
for path, dir_list, file_list in g:
    for file_name in file_list:
        fname = path + "/" + file_name
        nid = file_name.split(".jpg")[0]
        if nid not in dic_nidinfo:
            continue
        all_image_paths.append(fname)
        title = dic_nidinfo[nid]
        #print(max_length)
        inputs = multimodal_model.tokenizer(title, return_tensors="tf", max_length=multimodal_model.max_length, padding='max_length', truncation=True)
        #print(inputs)
        x1 = inputs["input_ids"]
        input_ids.append(tf.reshape(x1, [multimodal_model.max_length]))
        x2 = inputs["attention_mask"]
        attention_masks.append(tf.reshape(x2, [multimodal_model.max_length]))
        all_nids.append(nid)


path_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_nids, input_ids, attention_masks))
image_ds = path_ds.map(multimodal_model.read_tensor_from_image_file_map, num_parallel_calls=10)

batchsize = 256
#batchsize = 1
dataset = image_ds.batch(batchsize)

ckpt = sys.argv[3]
prefix = os.environ["flag"]

##multimodal_model1 = multimodal_model.multimodal(1, run_type="infer")
multimodal_model1 = multimodal_model.multimodal(1, run_type="infer_only_img")
#checkpoint_path = "models_ckpt/multimodal-{epoch:04d}.ckpt"
checkpoint_path = "models_ckpt/%s-%s.ckpt" % (prefix, ckpt)
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest = "models_ckpt/multimodal-0004.ckpt"
latest = checkpoint_path
multimodal_model1.load_weights(latest)
#multimodal_model2 = multimodal_model.multimodal(2)
#multimodal_cross = multimodal_model.cross_layer(multimodal_model1, multimodal_model2)


### predict

with open(file_out, 'w') as fout:
    for run_images, run_nid, run_input_ids, run_attention_masks in dataset:
        #print(run_images, run_input_ids, run_attention_masks)
        ##result = multimodal_model1.predict([run_input_ids, run_attention_masks, run_images])
        #result = multimodal_model1([run_input_ids, run_attention_masks, run_images], training=False) ## speedup
        result = multimodal_model1([run_images], training=False) ## speedup
        #result = multimodal_model1([run_input_ids, run_attention_masks], training=False) ## speedup only bert
        #result = multimodal_cross.predict([run_input_ids, run_attention_masks, run_images, run_input_ids, run_attention_masks, run_images])
        xresult = tf.reshape(result, [-1])
        res = tf.reshape(result, [-1])
        total_len = res.shape[0]
        print(total_len)
        for idx in range(0, total_len // multimodal_model.dim):
            xres = res[idx * multimodal_model.dim: idx * multimodal_model.dim + multimodal_model.dim].numpy()
            xxnid = run_nid[idx].numpy().decode()
            #print xxnid, 'qqq'
            out_vec = " ".join(str(i) for i in xres)
            out_str = "\t".join([str(xxnid), out_vec]) + "\n"
            fout.write(out_str)

