#!/bin/bash
set -x

#FLAG=off
#FLAG=on use export outside
#date=20220910 use env var outside
#date=20220910

date=`date -d "2 day ago" +"%Y%m%d"`
date=`date -d "1 day ago" +"%Y%m%d"`

hadoop_prefix="hdfs://haruna"
hadoop_cmd="hadoop"
python3=python3
python2=python2

export model_name=distiluse-base-multilingual-cased-v2
export vec_dim=512

cur_query_file=cur_query.$date
cur_first_name=cur_first_name.$date
cur_second_name=cur_second_name.$date
cur_third_name=cur_third_name.$date
user_query_file=user_query.$date
if [[ x$FLAG == x"off" ]]; then
    split_lines=10000
    date=20231229
    export g_per_query_res_min_cnt=20
    export g_cos_threshold=0.8
    export query_batch_size=100
    export pre_tag=test
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

function get_query()
{
    if [[ x$FLAG == x"on" ]]; then
        $hadoop_cmd fs -cat $hadoop_prefix/all_query_gip_all/date=$date/part* > $cur_query_file &
        $hadoop_cmd fs -cat $hadoop_prefix/first_name_new/date=$date/part* > $cur_first_name &
        $hadoop_cmd fs -cat $hadoop_prefix/second_name_new/date=$date/part* > $cur_second_name &
        $hadoop_cmd fs -cat $hadoop_prefix/third_name_new/date=$date/part* > $cur_third_name &
        $hadoop_cmd fs -cat $hadoop_prefix/all_cates_raw/date=$date/* > ./all_cates &
        # $hadoop_cmd fs -cat $hadoop_prefix/user_gip_query_history/date=$date/part* > $user_query_file &
        wait
        python3 ./gen_cates.py ./all_cates
    else
        cp $cur_query_file.head $cur_query_file
    fi

    # grep -v '\\N' $cur_first_name | awk -F'\1' '{print $1}' > first.txt    
    # grep -v '\\N' $cur_second_name | awk -F'\1' '{print $1}' > second.txt    
    # grep -v '\\N' $cur_third_name | awk -F'\1' '{print $1}' > third.txt    
    python gen_cates.py ./all_cates

}

function parallel_get_sim_res()
{
    query_file=$1
    author_file=$2
    prefix=$3
    all_file=$4
    rm $all_file.*
    $python3 -u get_sentence_bert_faiss.py $author_file ${query_file} ./latest_model $all_file

    bash split_file.sh $all_file $split_lines $all_file.indices
    for idx in `cat $all_file.indices`
    do
    {
        write_redis_parallel ./$all_file$idx $prefix
    } &
    done
    wait
    cat ./$all_file*.save.$prefix > ./$all_file.save.$prefix
    wc -l ./$all_file*
    xtarget_path=$hadoop_prefix/recall_res/${prefix}recall_res/
    $hadoop_cmd fs -mkdir $xtarget_path
    target_path=$xtarget_path/date=$date/
    $hadoop_cmd fs -rm -r $target_path
    $hadoop_cmd fs -mkdir $target_path
    $hadoop_cmd fs -put ./$all_file.save.$prefix $target_path
    $hadoop_cmd fs -put ./$all_file $target_path ## need merge first

}

function write_redis_parallel()
{
    filein=$1
    prefix=$2

    if [[ x$to_redis == x"no" ]]; then
        echo "not write redis"
    else
        $python2 write_query.py $filein $prefix 
    fi
    ls -lrt

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
    xtag=${cate_type}_${pre_tag}
    parallel_get_sim_res $cur_query_file $author_file.vec sbs_${xtag} recall_res_${xtag}.$date 
}


function main_f()
{
    get_latest_model
    get_query 
    get_cate_vec first.txt_mapping
    get_cate_vec second.txt_mapping
    get_cate_vec third.txt_mapping
    get_recall_res first.txt_mapping cate1 &
    get_recall_res second.txt_mapping cate2 & 
    get_recall_res third.txt_mapping cate3 &
    wait
    return 0
}

main_f 2>&1 ##> ./log/main.log.$date 2>&1 

