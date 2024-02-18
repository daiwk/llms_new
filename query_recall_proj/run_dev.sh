#!/bin/bash
set -x

#FLAG=off
#FLAG=on use export outside

date=dev

hadoop_prefix="hdfs://"
hadoop_cmd="hadoop"
python3=python3
python2=python2


cur_query_file=dev.query.txt



if [[ x$FLAG == x"off" ]]; then
    split_lines=10000
    date=20241229
    export g_per_query_res_min_cnt=20
    export g_cos_threshold=0.8
    export query_batch_size=100
    export pre_tag=test
    export model_name=distiluse-base-multilingual-cased-v2
    export vec_dim=512
else
    split_lines=1000000
    # export g_per_query_res_min_cnt=5
    # export g_cos_threshold=0.3
    # export query_batch_size=4096
    pip3 install -r ./requirements.txt
fi

function get_latest_model()
{
    if [[ x$FLAG == x"on" ]]; then
        rm -rf ./latest_model
        $hadoop_cmd fs -get $hadoop_prefix/Latest_models/$model_name ./latest_model
    fi

}

function parallel_get_sim_res()
{
    query_file=$1
    author_file=$2
    prefix=$3
    all_file=$4
    rm $all_file.*
    $python3 -u get_sentence_bert_faiss.py $author_file ${query_file} ./latest_model $all_file

    $python3 ./trans_format.py ./$all_file
    xtarget_path=$hadoop_prefix/dev_recall_res/
    $hadoop_cmd fs -mkdir $xtarget_path
    $hadoop_cmd fs -rmr $target_path/$all_file.save 
    $hadoop_cmd fs -put ./$all_file.save $target_path

}

function get_cate_vec()
{
    all_file=$1
    $python3 -u main_sentence_bert.py ./latest_model $all_file
}

function get_recall_res()
{
    author_file=$1
    cate_type=$2
    xtag=${model_name}_${cate_type}_${pre_tag}
    parallel_get_sim_res $cur_query_file $author_file.vec sbs_${xtag} recall_res_${xtag}.$date 
}


function main_f()
{
    cp ./dev_dir/* .
    get_latest_model
    get_cate_vec first.txt
    get_cate_vec second.txt
    get_cate_vec third.txt
    get_recall_res first.txt cate1 &
    get_recall_res second.txt cate2 &
    get_recall_res third.txt cate3 &
    wait
    return 0
}

main_f 2>&1 ##> ./log/main.log.$date 2>&1 

