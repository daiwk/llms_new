export CUR_BERT_BASE_DIR=.


function extract_feature()
{
    input_file=$1
    json_file=$2
    ./python-2.7.14/python-2.7.14/bin/python ./bin/bert/extract_features_bert_hadoop.py \
        --vocab_file=$CUR_BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$CUR_BERT_BASE_DIR/bert_config.json \
        --init_checkpoint=$CUR_BERT_BASE_DIR/bert_model.ckpt \
        --layers=-1 \
        --max_seq_length=128 \
        --batch_size=128
}

extract_feature 
