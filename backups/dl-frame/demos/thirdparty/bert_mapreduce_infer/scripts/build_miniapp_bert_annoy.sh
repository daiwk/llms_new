
input_file=$1
annoy_file=$2
idx_file=$3
vec_dim=$4

function build_miniapp_bert_annoy()
{
    input_file=$1
    annoy_file=$2
    idx_file=$3
    vec_dim=$4
    
    
    ./python-2.7.14/bin/python ./bin/bert/build_bert_annoy.py \
        --input_file=$input_file \
        --annoy_file=$annoy_file \
        --idx_file=$idx_file \
        --vec_dim=${vec_dim} \
        --use_layer=-1
}

build_miniapp_bert_annoy $input_file $annoy_file $idx_file $vec_dim
