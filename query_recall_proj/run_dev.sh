#!/bin/bash
set -x

#FLAG=off
#FLAG=on use export outside

date=dev

hadoop_prefix="hdfs://haruna"
hadoop_cmd="hadoop"
python3=python3
python2=python2


cur_query_file=dev.query.txt
cur_query_ecom_file=dev.query_ecom.txt
cur_query_not_ecom_file=dev.query_not_ecom.txt



if [[ x$FLAG == x"off" ]]; then
    split_lines=10000
    date=20231229
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

    $python3 ./trans_format_full.py ./$all_file ./$query_file
    xtarget_path=$hadoop_prefix/dev_recall_res/
    $hadoop_cmd fs -mkdir $xtarget_path
    $hadoop_cmd fs -rmr $xtarget_path/$all_file.save 
    $hadoop_cmd fs -put ./$all_file.save $xtarget_path

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
	query_file=$3
    xtag=${model_name}_${cate_type}_${query_file}
    parallel_get_sim_res $query_file $author_file.vec sbs_${xtag} recall_res_${xtag}.$date 
}


function main_f()
{
    cp ./dev_dir/* .
    get_latest_model
    get_cate_vec first.txt_mapping
    get_cate_vec second.txt_mapping
    get_cate_vec third.txt_mapping
    get_recall_res first.txt_mapping  cate1 $cur_query_ecom_file &
    get_recall_res second.txt_mapping cate2 $cur_query_ecom_file &
    get_recall_res third.txt_mapping  cate3 $cur_query_ecom_file &

    get_recall_res first.txt_mapping  cate1 $cur_query_not_ecom_file &
    get_recall_res second.txt_mapping cate2 $cur_query_not_ecom_file &
    get_recall_res third.txt_mapping  cate3 $cur_query_not_ecom_file &

    wait
    return 0
}

main_f 2>&1 ##> ./log/main.log.$date 2>&1 

