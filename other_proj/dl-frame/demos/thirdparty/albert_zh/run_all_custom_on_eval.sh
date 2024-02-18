#!/usr/bin/env bash
source ~/.bashrc

source ./source.conf
source ./source_on.conf
source ./source_eval.conf

export BERT_BASE_DIR=models_pretrain/albert_tiny_489k
export BERT_BASE_DIR=models_pretrain/albert_tiny_250k

export TEXT_DIR=./my_set
export MODEL_DIR=./albert_my_set_checkpoints
export PRE_MODEL_DIR=./albert_my_set_checkpoints_pre
## demo
start_finetune_step=900
start_predict_step=62

## on
start_finetune_step=8000
start_predict_step=42000

filelists=lst.pretrain.txt

function gen_ins_no_finetune()
{
    rm -rf $PRE_MODEL_DIR/*
    rm -rf $MODEL_DIR/*

    cd $TEXT_DIR
##    rm text.demo
##    hdxt fs -cat $hdfs_file > ./text.demo
    rm -rf ./pretrain.txt*
    time $python2 -u gen_pretrain_ins.py > pretrain.txt
    cd -
    sh -x split_file.sh $TEXT_DIR/pretrain.txt 200000 $filelists
}

function gen_ins_finetune()
{
##    rm -rf $PRE_MODEL_DIR/*
##    rm -rf $MODEL_DIR/*

    cd $TEXT_DIR
    time $python2 -u gen_finetune_ins.py
    cd -
}

function create_data_custom()
{

    for idx in `cat $filelists`
    do
    {
        $python create_pretraining_data.py \
            --do_whole_word_mask=True \
            --input_file=$TEXT_DIR/pretrain.txt$idx \
            --output_file=$PRE_MODEL_DIR/pre-train.tf_record.$idx \
            --vocab_file=$BERT_BASE_DIR/vocab.txt \
            --do_lower_case=True \
            --max_seq_length=128 \
            --max_predictions_per_seq=128 \
            --masked_lm_prob=0.10
    } &
    done
    wait

}

function pretrain_custom()
{
    file_list=""
    for idx in `cat $filelists` 
    do
        file_list=$PRE_MODEL_DIR/pre-train.tf_record.$idx,$file_list
    done

    ## 如果要从头pretrain，那就注释掉init_checkpoint

    $python run_pretraining.py \
            --input_file=$file_list \
            --output_dir=$PRE_MODEL_DIR \
            --do_train=True \
            --do_eval=True \
            --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
            --train_batch_size=32 \
            --max_seq_length=128 \
            --max_predictions_per_seq=128 \
            --learning_rate=0.00176 \
            --num_train_steps=100000 \
            --num_warmup_steps=1000 \
            --save_checkpoints_steps=2000   \
            --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt 

    #        --num_train_steps=10000 \
    #        --num_warmup_steps=1000 \
    #        --save_checkpoints_steps=2000   \
}

function board()
{
    model=$1 
    port=$2
    ps aux|grep tensorboard |grep $port | awk '{print $2}'| xargs kill -9 
    nohup $tensorboard --logdir=./$model/ --port=$port --host=`hostname` &
}

function finetune_custom()
{
    rm $MODEL_DIR/train.tf_record
    start_finetune_step=`ls -lrt $PRE_MODEL_DIR/model.ckpt* -lrt| tail -n 1 | awk -F'model.ckpt-' '{print $2}'| awk -F'.' '{print $1}'`
    $python run_classifier.py   \
        --task_name=lcqmc_pair   \
        --do_train=true   \
        --do_eval=true   \
        --data_dir=$TEXT_DIR   \
        --vocab_file=$BERT_BASE_DIR/vocab.txt  \
        --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
        --max_seq_length=128 \
        --train_batch_size=64   \
        --learning_rate=1e-4 \
        --num_train_epochs=1 \
        --output_dir=$MODEL_DIR \
        --init_checkpoint=$PRE_MODEL_DIR/model.ckpt-$start_finetune_step

}

function finetune_custom_from_raw()
{
    rm $MODEL_DIR/train.tf_record
    $python run_classifier.py   \
        --task_name=lcqmc_pair   \
        --do_train=true   \
        --do_eval=true   \
        --data_dir=$TEXT_DIR   \
        --vocab_file=$BERT_BASE_DIR/vocab.txt  \
        --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
        --max_seq_length=128 \
        --train_batch_size=64   \
        --learning_rate=1e-4 \
        --num_train_epochs=2 \
        --output_dir=$MODEL_DIR \
        --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt 

}

function predict_custom_from_pretrain()
{
    cd $TEXT_DIR
    python $gen_predict_ins
    cd -
    # must run train in finetune first, then use its output_dir
    start_predict_step=`ls -lrt $PRE_MODEL_DIR/model.ckpt* -lrt| tail -n 1 | awk -F'model.ckpt-' '{print $2}'| awk -F'.' '{print $1}'`
    $python run_classifier.py   \
        --task_name=lcqmc_pair \
        --do_predict=true \
        --data_dir=$TEXT_DIR   \
        --vocab_file=$BERT_BASE_DIR/vocab.txt  \
        --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
        --max_seq_length=128 \
        --output_dir=$MODEL_DIR \
        --predict_batch_size=1 \
        --init_checkpoint=$PRE_MODEL_DIR/model.ckpt-$start_predict_step

##        --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt 
    cd $TEXT_DIR
    python parse_res.py
}

function predict_custom_from_finetune()
{
    cd $TEXT_DIR
    python $gen_predict_ins
    cd -
    # must run train in finetune first, then use its output_dir
    start_predict_step=`ls -lrt $MODEL_DIR/model.ckpt* -lrt| tail -n 1 | awk -F'model.ckpt-' '{print $2}'| awk -F'.' '{print $1}'`
    $python run_classifier.py   \
        --task_name=lcqmc_pair \
        --do_predict=true \
        --data_dir=$TEXT_DIR   \
        --vocab_file=$BERT_BASE_DIR/vocab.txt  \
        --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
        --max_seq_length=128 \
        --output_dir=$MODEL_DIR \
        --predict_batch_size=100 \
        --init_checkpoint=$MODEL_DIR/model.ckpt-$start_predict_step

##        --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt 
    cd $TEXT_DIR
    python parse_res.py
    cd -
}

function only_pretrain()
{
    ## only pretrain
    gen_ins_no_finetune
    create_data_custom
    pretrain_custom
    predict_custom_from_pretrain
}

function pretrain_and_finetune()
{
    ## pretrain + finetune
    gen_ins_no_finetune
    create_data_custom
    pretrain_custom
    gen_ins_finetune
    finetune_custom
    predict_custom_from_pretrain
}

function only_finetune()
{
    ## only finetune
##    gen_ins_finetune
##    finetune_custom_from_raw
    predict_custom_from_finetune
}

function only_extract()
{
    cd $TEXT_DIR
    python $gen_extract_ins
    cd -
    # must run train in finetune first, then use its output_dir
    start_predict_step=`ls -lrt $MODEL_DIR/model.ckpt* -lrt| tail -n 1 | awk -F'model.ckpt-' '{print $2}'| awk -F'.' '{print $1}'`
    $python extract_features_albert.py   \
        --task_name=lcqmc_pair \
        --do_predict=true \
        --layers=-1 \
        --data_dir=$TEXT_DIR   \
        --vocab_file=$BERT_BASE_DIR/vocab.txt  \
        --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
        --max_seq_length=128 \
        --output_dir=$MODEL_DIR \
        --predict_batch_size=100 \
        --init_checkpoint=$MODEL_DIR/model.ckpt-$start_predict_step

##        --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt 
#    cd $TEXT_DIR
#    python parse_res.py

}

function main()
{
    board $MODEL_DIR 8001
    board $PRE_MODEL_DIR 8002

#    pretrain_and_finetune
#    only_pretrain
    time only_finetune
    cd eval_dir
    python eval_x.py  ## cur best:albert_x_med_len25_att_top3cat1_2_nozonghe 0.116653292334 370 199.445945946
#    only_extract

}

main >log/run_custom_on_eval.log 2>&1
