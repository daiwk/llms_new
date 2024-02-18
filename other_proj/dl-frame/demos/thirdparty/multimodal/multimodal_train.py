#encoding=utf8
import datetime
import sys
import tensorflow as tf
import tensorflow_addons as tfa
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

width = int(os.environ["WIDTH"])
height = int(os.environ["HEIGHT"])


#bad_nids = set()
#with open("./badcase.txt", 'r') as finbad:
#    for line in finbad:
#        nid = line.strip("\t")[0]
#        bad_nids.add(nid)


black_mouth_nid = set()
with open("./black_mouth_nid.txt", 'r') as finblack_mouth:
    for line in finblack_mouth:
        nid = line.strip("\t")[0]
        black_mouth_nid.add(nid)


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

input_dir = "./imgs/"

input_dir = sys.argv[1]


dic_nidinfo = {}
dic_label = {}
xidx = 0
label2id = {}
with open("label.dict", 'r') as finx:
    for line in finx:
        line = line.strip("\n")
        label2id[line] = xidx
        xidx += 1
if "infer" not in input_dir:
    nid_info = "info.res"
    with open(nid_info, 'r') as fin_nidinfo:
        for line in fin_nidinfo:
            line = line.strip("\n").split("\t")
            if len(line) != 6:
                continue
            nid = line[0]
            if nid in black_mouth_nid:
                continue
            #title = line[3][:100] # title
            cate = line[2] # cate
            title = line[3] # title
            dic_nidinfo[nid] = title
            dic_label[nid] = label2id[cate]
            
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

write_ds = sys.argv[2]
images_count = 0
g = os.walk(input_dir)
for path, dir_list, file_list in g:
    for file_name in file_list:
        fname = path + "/" + file_name
        nid = file_name.split(".jpg")[0]
        if nid not in dic_nidinfo:
            continue
        images_count += 1

if write_ds == "write_ds":

    all_image_paths = []
    all_nids = []
    all_labels = []
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
            label = dic_label[nid]
            all_labels.append(label)

    path_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_nids, input_ids, attention_masks, all_labels))
    image_ds = path_ds.map(multimodal_model.read_tensor_from_image_file_map_train, num_parallel_calls=10)
    print("image_ds:", image_ds)
    # image_ds: <ParallelMapDataset shapes: ((224, 224, 3), (), (64,), (64,), ()), types: (tf.uint8, tf.string, tf.int32, tf.int32, tf.int32)>
    img_ds = image_ds.map(lambda a, b, c, d, e: tf.io.serialize_tensor(a))
    tfrec = tf.data.experimental.TFRecordWriter('train_img.tfrec')
    tfrec.write(img_ds)
    nid_ds = image_ds.map(lambda a, b, c, d, e: tf.io.serialize_tensor(b))
    tfrec = tf.data.experimental.TFRecordWriter('train_nid.tfrec')
    tfrec.write(nid_ds)
    input_id_ds= image_ds.map(lambda a, b, c, d, e: tf.io.serialize_tensor(c))
    tfrec = tf.data.experimental.TFRecordWriter('train_input_id.tfrec')
    tfrec.write(input_id_ds)
    attention_mask_ds = image_ds.map(lambda a, b, c, d, e: tf.io.serialize_tensor(d))
    tfrec = tf.data.experimental.TFRecordWriter('train_attention_mask.tfrec')
    tfrec.write(attention_mask_ds)
    label_ds = image_ds.map(lambda a, b, c, d, e: tf.io.serialize_tensor(e))
    tfrec = tf.data.experimental.TFRecordWriter('train_label.tfrec')
    tfrec.write(label_ds)
    exit(0)
else:
    def parse_img(x):
        result = tf.io.parse_tensor(x, out_type=tf.uint8)
        result = tf.reshape(result, [width, height, 3])
        return result
    img_ds = tf.data.TFRecordDataset('train_img.tfrec').map(parse_img, num_parallel_calls=tf.data.AUTOTUNE)
    def parse_nid(x):
        result = tf.io.parse_tensor(x, out_type=tf.string)
        result = tf.reshape(result, [])
        return result
    nid_ds = tf.data.TFRecordDataset('train_nid.tfrec').map(parse_nid, num_parallel_calls=tf.data.AUTOTUNE)
    def parse_input_id(x):
        result = tf.io.parse_tensor(x, out_type=tf.int32)
        result = tf.reshape(result, [64,])
        return result
    input_id_ds = tf.data.TFRecordDataset('train_input_id.tfrec').map(parse_input_id, num_parallel_calls=tf.data.AUTOTUNE)
    def parse_attention_mask(x):
        result = tf.io.parse_tensor(x, out_type=tf.int32)
        result = tf.reshape(result, [64,])
        return result
    attention_mask_ds = tf.data.TFRecordDataset('train_attention_mask.tfrec').map(parse_attention_mask, num_parallel_calls=tf.data.AUTOTUNE)
    def parse_label(x):
        result = tf.io.parse_tensor(x, out_type=tf.int32)
        result = tf.reshape(result, [])
        return result
    label_ds = tf.data.TFRecordDataset('train_label.tfrec').map(parse_label, num_parallel_calls=tf.data.AUTOTUNE)
    image_ds = tf.data.Dataset.zip((img_ds, nid_ds, input_id_ds, attention_mask_ds, label_ds))
    print(image_ds)
    print(img_ds)
    print(nid_ds)
    print(input_id_ds)
    print(attention_mask_ds)
    print(label_ds)
    

batchsize = 16
#batchsize = 256
#batchsize = 1
#dataset = image_ds.batch(batchsize)


#multimodal_model_classify = multimodal_model.multimodal(1, run_type="train")
multimodal_model_classify = multimodal_model.multimodal(1, run_type="train_only_img")
#multimodal_model2 = multimodal_model.multimodal(2)
#multimodal_cross = multimodal_model.cross_layer(multimodal_model1, multimodal_model2)

#model = multimodal_cross
model = multimodal_model_classify

data_aug = "no_data_augmentation"

train_examples = 10000
buffer_size = train_examples

options = tf.data.Options()
options.experimental_threading.max_intra_op_parallelism = 1
image_ds = tf.data.Dataset.zip((img_ds, label_ds))
dataset = image_ds.with_options(options)

dataset = dataset.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=train_examples)).batch(batchsize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

top5_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5acc")
top10_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="top10acc")
learning_rate=1e-4
learning_rate=1e-2
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy", top5_acc, top10_acc])

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
prefix = "%s-%d-%f" % (data_aug, batchsize, learning_rate)
prefix = os.environ["flag"]
train_log_dir = "logs/%s" % prefix + "/train"
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=train_log_dir,
    histogram_freq=0,  # How often to log histogram visualizations
    embeddings_freq=0,  # How often to log embedding visualizations
    update_freq="batch",
)  # How often to write logs (default: once per epoch)

# 在文件名中包含 epoch (使用 `str.format`)
checkpoint_path = "models_ckpt/%s-{epoch:04d}.ckpt" % (prefix)
#checkpoint_dir = os.path.dirname(checkpoint_path)


save_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    verbose=1,
    monitor='val_accuracy',
    mode='max',
    save_best_only=False)

#model.fit(dataset, epochs=1, steps_per_epoch=3, callbacks=[tb_callback])
steps_per_epoch = tf.math.ceil(images_count / batchsize).numpy()
print(steps_per_epoch)
num_epochs = 30
model.fit(dataset, epochs=num_epochs, callbacks=[tb_callback, save_callback], steps_per_epoch=steps_per_epoch)

exit(0)

####optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
###optimizer = tfa.optimizers.LAMB(learning_rate=0.01)
###loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
###
###def loss(model, x, y):
###  y_ = model(x)
###  return loss_object(y_true=y, y_pred=y_)
###
###train_loss_results = []
###train_accuracy_results = []
###
###
##### 创建一个回调，每 5 个 epochs 保存模型的权重
####cp_callback = tf.keras.callbacks.ModelCheckpoint(
####    filepath=checkpoint_path, 
####    verbose=1, 
####    save_weights_only=True,
####    period=5)
###
####l = loss(model, features, labels)
###
###def grad(model, inputs, targets):
###  with tf.GradientTape() as tape:
###    loss_value = loss(model, inputs, targets)
###  return loss_value, tape.gradient(loss_value, model.trainable_variables)
###
###current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
###train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
###train_summary_writer = tf.summary.create_file_writer(train_log_dir)
###
###
###
###
###for epoch in range(num_epochs):
###  epoch_loss_avg = tf.keras.metrics.Mean()
###  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
###  # Training loop - using batches of 32
###  for run_images, run_nid, run_input_ids, run_attention_masks, label in dataset:
###    # 优化模型
###    #x = [run_input_ids, run_attention_masks, run_images]
###    x = [run_images]
###    y = label
###    loss_value, grads = grad(model, x, y)
###    optimizer.apply_gradients(zip(grads, model.trainable_variables))
###
###    # 追踪进度
###    epoch_loss_avg(loss_value)  # 添加当前的 batch loss
###    # 比较预测标签与真实标签
###    epoch_accuracy(y, model(x))
###
###  with train_summary_writer.as_default():
###    tf.summary.scalar('loss', epoch_loss_avg.result(), step=epoch)
###    tf.summary.scalar('accuracy', epoch_accuracy.result(), step=epoch)
###  # 循环结束
###  train_loss_results.append(epoch_loss_avg.result())
###  train_accuracy_results.append(epoch_accuracy.result())
###
###  if epoch % 2 == 0:
###    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
###                                                                epoch_loss_avg.result(),
###                                                                epoch_accuracy.result()))
###
###    model.save_weights(checkpoint_path.format(epoch=epoch))
###
###
######for run_images, run_nid, run_input_ids, run_attention_masks in dataset:
######    #print(run_images, run_input_ids, run_attention_masks)
######    ##result = multimodal_model1.predict([run_input_ids, run_attention_masks, run_images])
######    #result = multimodal_model1([run_input_ids, run_attention_masks, run_images], training=False) ## speedup
######    result = multimodal_model1([run_input_ids, run_attention_masks], training=False) ## speedup only bert
######    #result = multimodal_cross.predict([run_input_ids, run_attention_masks, run_images, run_input_ids, run_attention_masks, run_images])
######    xresult = tf.reshape(result, [-1])
######    res = tf.reshape(result, [-1])
######    total_len = res.shape[0]
######    print(total_len)
######    for idx in range(0, total_len // multimodal_model.dim):
######        xres = res[idx * multimodal_model.dim: idx * multimodal_model.dim + multimodal_model.dim].numpy()
######        xxnid = run_nid[idx].numpy().decode()
######        #print xxnid, 'qqq'
######        out_vec = " ".join(str(i) for i in xres)
######        out_str = "\t".join([str(xxnid), out_vec]) + "\n"
######        fout.write(out_str)
######
