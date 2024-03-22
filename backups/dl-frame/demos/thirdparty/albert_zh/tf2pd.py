import tensorflow as tf
from tensorflow.python.framework import graph_util
import sys

def freeze_model(sess, output_tensor_names, freeze_model_path):
    out_graph = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), output_tensor_names)
    with tf.gfile.GFile(freeze_model_path, 'wb') as f:
        f.write(out_graph.SerializeToString())

    print("freeze model saved in {}".format(freeze_model_path))

sess = tf.Session()
#saver = tf.train.import_meta_graph('albert_xlarge_zh_183k/albert_model.ckpt.meta')

step = 14543

saver = tf.train.import_meta_graph('albert_my_set_checkpoints/model.ckpt-{}.meta'.format(step))
saver.restore(sess, tf.train.latest_checkpoint('albert_my_set_checkpoints'))


x = [n.name for n in tf.get_default_graph().as_graph_def().node]
for a in x:
    print(a)

#saver = tf.train.import_meta_graph('models_pretrain/albert_tiny_250k/albert_model.ckpt.meta')
#saver.restore(sess, tf.train.latest_checkpoint('models_pretrain/albert_tiny_250k'))
#output_names = ['bert/pooler/dense/Tanh']
output_names = ['bert/pooler/dense/Tanh', 'loss/Softmax']
freeze_model(sess, output_names, './albert_dwk.pb')
