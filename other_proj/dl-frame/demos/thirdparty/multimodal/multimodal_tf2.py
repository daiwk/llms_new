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
max_length = 32
max_length = 64

bert_type = "albert-base-v2" # slow, 12layers
bert_type = "distilbert-base-uncased" # 6layers

if bert_type == "distilbert-base-uncased":
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    raw_transformer_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
elif bert_type == "albert-base-v2":
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    raw_transformer_model = TFAlbertModel.from_pretrained('albert-base-v2')

initializer = tf.keras.initializers.HeNormal()

istraining = False

def add_prefix(model, prefix: str, custom_objects=None):
    """Adds a prefix to layers and model name while keeping the pre-trained weights
    Arguments:
        model: a tf.keras model
        prefix: a string that would be added to before each layer name
        custom_objects: if your model consists of custom layers you shoud add them pass them as a dictionary. 
            For more information read the following:
            https://keras.io/guides/serialization_and_saving/#custom-objects
    Returns:
        new_model: a tf.keras model having same weights as the input model.
    """
    
    config = model.get_config()
    old_to_new = {}
    new_to_old = {}
    
    for layer in config['layers']:
        new_name = prefix + layer['name']
        old_to_new[layer['name']], new_to_old[new_name] = new_name, layer['name']
        layer['name'] = new_name
        layer['config']['name'] = new_name

        if len(layer['inbound_nodes']) > 0:
            for in_node in layer['inbound_nodes'][0]:
                in_node[0] = old_to_new[in_node[0]]
    
    for input_layer in config['input_layers']:
        input_layer[0] = old_to_new[input_layer[0]]
    
    for output_layer in config['output_layers']:
        output_layer[0] = old_to_new[output_layer[0]]
    
    config['name'] = prefix + config['name']
    new_model = tf.keras.Model().from_config(config, custom_objects)
    
    for layer in new_model.layers:
        layer.set_weights(model.get_layer(new_to_old[layer.name]).get_weights())
    
    return new_model 


def multimodal(model_num):
    """multimodal"""
    base_model = image_model(model_num)
    # Get pretrained language model with transformer architecture
    #raw_transformer_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    #transformer_model = add_prefix(raw_transformer_model, "%s_%d_" % (model_type, model_num))
    transformer_model = raw_transformer_model
#    for param in transformer_model.named_parameters():
#        param.requires_grad = False

    for layer in transformer_model.layers:
        layer.trainable= False

    # define input layers for distilbert
    input_ids_in = tf.keras.layers.Input(shape=(max_length,), name='input_token_%d' % model_num, dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(max_length,), name='masked_token_%d' % model_num, dtype='int32')

    # extract embedding layer
    embedding_layer = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]
    cls_token = embedding_layer[:,0,:]

    # add feed forward layers
    language_model = tf.keras.layers.BatchNormalization()(cls_token)
    language_model = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer)(language_model)
#    language_model = tf.keras.layers.Dropout(0.2)(language_model)
#    language_model = tf.keras.layers.Dense(64, activation='relu')(language_model)
#    language_model = tf.keras.layers.Dense(32, activation='relu')(language_model)

    # Add numerical layer
    #numerical_input = tf.keras.layers.Input(shape=(numerical_input.shape[1],), name='numerical_input', dtype='float64')
    numerical_layer = tf.keras.layers.Dense(128, input_dim=base_model.output.shape[1], activation='relu', kernel_initializer=initializer)(base_model.output)
    numerical_layer = tf.keras.layers.Flatten()(numerical_layer)

    # Concatenate both layers
    concatted = tf.keras.layers.Concatenate()([language_model, numerical_layer])

    # Add classification head with sigmoid activation
    head = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer)(concatted)
    ##head = tf.keras.layers.Dense(y_labels.shape[1], activation='sigmoid')(head)

    # define model with three input types, first two types will come from the tokenizer
    ##multimodal_model = tf.keras.Model(inputs=[input_ids_in, input_masks_in, base_model.input], outputs = head, name='Multimodal_%d' % model_num)
    
    # only bert
    multimodal_model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs=cls_token, name='Multimodal_%d' % model_num)

    # Prevent Distilbert from being trainable
    #multimodal_model.layers[2].trainable = False
    multimodal_model.summary()
    return multimodal_model


def cross_layer(multimodal_model1, multimodal_model2):
    """cross_layer"""
    #model.compile(optimizer='sgd', loss=tf.keras.losses.CosineSimilarity(axis=1))
    ###cos_uv = tf.keras.losses.cosine_similarity(multimodal_model1.output, multimodal_model2.output)
    ###output = cos_uv
    dot_uv = tf.reduce_sum(multimodal_model1.output * multimodal_model2.output, axis=1)
    dot_uv = tf.expand_dims(dot_uv, 1)
    output = tf.keras.activations.sigmoid(dot_uv)
    #output = tf.keras.layers.Dense(1, activation='sigmoid')(dot_uv)
    #output = dot_uv
    
    res_model = tf.keras.models.Model(inputs=[multimodal_model1.input, multimodal_model2.input], outputs=[output])
    res_model.summary()
    print(res_model.trainable_weights)
    tf.keras.utils.plot_model(res_model, to_file='multimodal_cross.png', show_shapes=True, show_layer_names=True, expand_nested=True)
    return res_model


width = int(os.environ["WIDTH"])
height = int(os.environ["HEIGHT"])
model_type = os.environ["MODEL_TYPE"]
dim = int(os.environ["DIM"])


def image_model(model_num):
    """image_model"""
    i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
    x = tf.cast(i, tf.float32)

    if model_type == "MobileNet":
        base_model = tf.keras.applications.MobileNet(weights='imagenet')
        #base_model._name = "MobileNet_%d" % model_num ## useless
        x = tf.keras.applications.mobilenet.preprocess_input(x)
        base_model.summary()
        model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('dropout').output, name="img_%d" % model_num) # 1024

##        for layer in model.layers:
##            layer.name = layer.name + str("_%d" % model_num) 

    elif model_type == "DenseNet201":
        base_model = tf.keras.applications.DenseNet201(weights='imagenet')
        x = tf.keras.applications.densenet.preprocess_input(x)
        base_model.summary()
        model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output) # 1920

    elif model_type == "NASNetMobile":
        base_model = tf.keras.applications.NASNetMobile(weights='imagenet')
        x = tf.keras.applications.nasnet.preprocess_input(x)
        base_model.summary()
        model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output) # 1056

    elif model_type == "InceptionResNetV2":
        base_model = tf.keras.applications.InceptionResNetV2(weights='imagenet') # input 299 299
        x = tf.keras.applications.inception_resnet_v2.preprocess_input(x)
        base_model.summary()
        model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output) # 1536

    elif model_type == "Xception":
        base_model = tf.keras.applications.Xception(weights='imagenet')# input 299 299
        x = tf.keras.applications.xception.preprocess_input(x)
        base_model.summary()
        model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output) # 2048

    new_model = add_prefix(model, "%s_%d_" % (model_type, model_num))
    for layer in new_model.layers:
        layer.trainable = False
    new_model.summary()

    return new_model


#x = core(x)
####model = tf.keras.Model(inputs=[i], outputs=[core.get_layer('add_15').output])
#model = tf.keras.Model(inputs=[i], outputs=[x])

bad_nids = set()
with open("./badcase.txt", 'r') as finbad:
    for line in finbad:
        nid = line.strip("\t")[0]
        bad_nids.add(nid)

def read_tensor_from_image_file_map(file_name, nid, input_ids, attention_masks):
    """read_tensor_from_image_file_map"""
    input_name = "file_reader"
    #print(file_name)
    file_reader = tf.io.read_file(file_name, input_name)
    image_reader = tf.io.decode_jpeg(
          file_reader, channels=3, name="jpeg_reader", dct_method='INTEGER_ACCURATE')
    #image_reader = tf.image.resize(image_reader, [224, 224])
    #image_reader = tf.image.resize(image_reader, [299, 299])
    image_reader = tf.image.resize(image_reader, [width, height])
    image_reader = tf.cast(image_reader, np.uint8)
    return image_reader, nid, input_ids, attention_masks
    #return resized, nid


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    """read_tensor_from_image_file"""
    input_name = "file_reader"
    output_name = "normalized"
    #print file_name
    file_reader = tf.io.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.io.decode_png(file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.io.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.io.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.io.decode_jpeg(
          file_reader, channels=3, name="jpeg_reader")
##    float_caster = tf.cast(image_reader, tf.float32)
##    dims_expander = tf.expand_dims(float_caster, 0)
##    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
##    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    return image_reader 

file_out = "img.vec"
input_dir = "./imgs/"

file_out = sys.argv[2]
input_dir = sys.argv[1]


dic_nidinfo = {}
if "infer" not in input_dir:
    nid_info = "info.res"
    with open(nid_info, 'r') as fin_nidinfo:
        for line in fin_nidinfo:
            line = line.strip("\n").split("\t")
            if len(line) != 6:
                continue
            nid = line[0]
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
        inputs = tokenizer(title, return_tensors="tf", max_length=max_length, padding='max_length', truncation=True)
        #print(inputs)
        x1 = inputs["input_ids"]
        input_ids.append(tf.reshape(x1, [max_length]))
        x2 = inputs["attention_mask"]
        attention_masks.append(tf.reshape(x2, [max_length]))
        all_nids.append(nid)


##path_ds_train = tf.data.Dataset.from_tensor_slices((all_image_paths_train, all_nids_train, input_ids_train, attention_masks_train, labels_train))
##image_ds_train = path_ds.map(read_tensor_from_image_file_map_train, num_parallel_calls=10)



path_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_nids, input_ids, attention_masks))
image_ds = path_ds.map(read_tensor_from_image_file_map, num_parallel_calls=10)

batchsize = 256
#batchsize = 1
dataset = image_ds.batch(batchsize)

multimodal_model1 = multimodal(1)
multimodal_model2 = multimodal(2)
multimodal_cross = cross_layer(multimodal_model1, multimodal_model2)


### predict

with open(file_out, 'w') as fout:
    for run_images, run_nid, run_input_ids, run_attention_masks in dataset:
        #print(run_images, run_input_ids, run_attention_masks)
        ##result = multimodal_model1.predict([run_input_ids, run_attention_masks, run_images])
        #result = multimodal_model1([run_input_ids, run_attention_masks, run_images], training=False) ## speedup
        result = multimodal_model1([run_input_ids, run_attention_masks], training=False) ## speedup only bert
        #result = multimodal_cross.predict([run_input_ids, run_attention_masks, run_images, run_input_ids, run_attention_masks, run_images])
        xresult = tf.reshape(result, [-1])
        res = tf.reshape(result, [-1])
        total_len = res.shape[0]
        print(total_len)
        for idx in range(0, total_len // dim):
            xres = res[idx * dim: idx * dim + dim].numpy()
            xxnid = run_nid[idx].numpy().decode()
            #print xxnid, 'qqq'
            out_vec = " ".join(str(i) for i in xres)
            out_str = "\t".join([str(xxnid), out_vec]) + "\n"
            fout.write(out_str)
