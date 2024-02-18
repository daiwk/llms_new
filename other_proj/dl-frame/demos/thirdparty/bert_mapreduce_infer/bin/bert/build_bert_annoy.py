#encoding=utf8


import tensorflow as tf
import json
from annoy import AnnoyIndex


flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")
flags.DEFINE_integer("vec_dim", None, 64)
flags.DEFINE_integer("use_layer", None, -1)
flags.DEFINE_string("annoy_file", None, "")
flags.DEFINE_string("idx_file", None, "")

def get_nid(linex_index):
    return linex_index

if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("use_layer")
    flags.mark_flag_as_required("vec_dim")
    flags.mark_flag_as_required("annoy_file")
    flags.mark_flag_as_required("idx_file")
    vec_dim = FLAGS.vec_dim
    annoy_lib = AnnoyIndex(vec_dim, metric='angular')
    chosen_token = "[CLS]"
    idx2nid_file = open(FLAGS.idx_file, "wb")
    annoy_file = FLAGS.annoy_file
    idx = 0
    with open(FLAGS.input_file, 'rb') as fin:
        for line in fin:
            line = line.strip("\n")
            js = json.loads(line)
            linex_index = js["linex_index"]
            for fea in js["features"]:
                if fea["token"] == chosen_token:
                    for layer in fea["layers"]:
                        if layer["index"] == FLAGS.use_layer:
                            vec = layer["values"]
                            nid = linex_index
                            idx2nid_file.write('%d\t%s\n' % (idx, nid))
                            annoy_lib.add_item(idx, vec)
                            idx += 1
    
  
    annoy_lib.build(10)
    annoy_lib.save(annoy_file)
