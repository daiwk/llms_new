#encoding=utf8
import sys
import tensorflow as tf
import os
import ssl
import numpy as np
import cv2
#from tensorflow.python.keras import backend as K
from keras import backend as K
#from keras.backend import manual_variable_initialization
##np.random.seed(1337) # for reproducibility
#manual_variable_initialization(True)

ssl._create_default_https_context = ssl._create_unverified_context
i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
x = tf.cast(i, tf.float32)
###x = tf.keras.applications.mobilenet.preprocess_input(x)

#base_model = tf.keras.applications.VGG16(weights='imagenet')
#x = tf.keras.applications.vgg16.preprocess_input(x)
#model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output) # 28 *28 *512
#dim = 401408

width = int(os.environ["WIDTH"])
height = int(os.environ["HEIGHT"])
model_type = os.environ["MODEL_TYPE"]
dim = int(os.environ["DIM"])

if model_type == "MobileNet":
    base_model = tf.keras.applications.MobileNet(weights='imagenet')
    x = tf.keras.applications.mobilenet.preprocess_input(x)
    base_model.summary()
    model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('dropout').output) # 1024

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


model.summary()
for layer in model.layers:
    layer.trainable = False
#x = core(x)
####model = tf.keras.Model(inputs=[i], outputs=[core.get_layer('add_15').output])
#model = tf.keras.Model(inputs=[i], outputs=[x])

bad_nids = set()
with open("./badcase.txt", 'rb') as finbad:
    for line in finbad:
        nid = line.strip("\t")[0]
        bad_nids.add(nid)

def read_tensor_from_image_file_map(file_name, nid):
    """read_tensor_from_image_file_map"""
    input_name = "file_reader"
    file_reader = tf.read_file(file_name, input_name)
    image_reader = tf.io.decode_jpeg(
          file_reader, channels=3, name="jpeg_reader", dct_method='INTEGER_ACCURATE')
    #image_reader = tf.image.resize(image_reader, [224, 224])
    #image_reader = tf.image.resize(image_reader, [299, 299])
    image_reader = tf.image.resize(image_reader, [width, height])
    image_reader = tf.cast(image_reader, np.uint8)
    return image_reader, nid
    #return resized, nid


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    """read_tensor_from_image_file"""
    input_name = "file_reader"
    output_name = "normalized"
    print file_name
    file_reader = tf.read_file(file_name, input_name)
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

# limit to num_cpu_core CPU usage 限制CPU使用的核的个数
config = tf.ConfigProto(device_count={"CPU": 20}, 
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0)
#        log_device_placement=True)
   
file_out = "img.vec"
input_dir = "./imgs/"

file_out = sys.argv[2]
input_dir = sys.argv[1]


all_image_paths = []
all_nids = []
g = os.walk(input_dir)
#g = os.walk("./imgs_test/")
for path, dir_list, file_list in g:
    for file_name in file_list:
        fname = path + "/" + file_name
        all_image_paths.append(fname)
        nid = file_name.split(".jpg")[0]
        print nid, 'xx'
        all_nids.append(nid)

path_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_nids))
image_ds = path_ds.map(read_tensor_from_image_file_map, num_parallel_calls=10)

batchsize = 256
#batchsize = 1
dataset = image_ds.batch(batchsize)

#iterator = dataset.make_one_shot_iterator()
iterator = dataset.make_initializable_iterator()
images, nids = iterator.get_next()

#with tf.Session(config=config) as session, \
with tf.keras.backend.get_session() as session, \
    open(file_out, 'wb') as fout:
    K.set_session(session)
    #session = tf.keras.backend.get_session()
    #init = tf.global_variables_initializer()
    #session.run(init)
    session.run(iterator.initializer)

    try:
        while True:
            run_images, run_nid = session.run([images, nids])
            result = model.predict(run_images)
            xresult = tf.reshape(result, [-1])
            res = session.run(xresult)
            total_len = res.shape[0]
            print total_len
            for idx in xrange(0, total_len / dim):
                xres = res[idx * dim: idx * dim + dim]
                xxnid = run_nid[idx]
                #print xxnid, 'qqq'
                out_vec = " ".join(str(i) for i in xres)
                out_str = "\t".join([xxnid, out_vec]) + "\n"
                fout.write(out_str)
                #exit(0)
    except tf.errors.OutOfRangeError:
        pass

##                image = read_tensor_from_image_file(fname)
##                image = tf.expand_dims(image, 0) 
##                resized = tf.image.resize_bilinear(image, [224, 224])
##                result = model(image)
##                xresult = tf.reshape(result, [-1])
##                res = session.run(xresult)
##                out_vec = " ".join(str(i) for i in res)
##                nid = file_name.split(".jpg")[0]
##                out_str = "\t".join([nid, out_vec]) + "\n"
##                fout.write(out_str)
