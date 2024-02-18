#!/bin/sh
set -x

source ~/.bashrc

JOB_PRI=HIGH

DATE=`date +"%Y%m%d"`
yesterday=`date +"%Y%m%d" -d "1 day ago"`
yesterday=`cat ./dict/yesterday`
DATE=`cat ./dict/today`

export use_official=1

#input_file=./dict/miniapp_title
#output_file=./dict/miniapp_title
input_file=$1
output_file=$2
tag=$3

function run()
{

INPUT=xxxx/${tag}_data/

hadoop fs -rmr $INPUT/*
hadoop fs -mkdir $INPUT/

hadoop fs -put $input_file $INPUT/

OUTPUT=xxxx/${tag}_title/${DATE}/

JOB_NAME="xxxx_${tag}_title_$DATE"

HADOOP_PYTHON_PATH=xxxx/python-2.7.14.tar.gz

if [[ ${use_official} -eq 0 ]];then
    BERT_BASE_DIR=xxxx/b_model/$yesterday/chinese_wordpiece_small
    model_step=`hadoop fs -cat ${BERT_BASE_DIR}/output/checkpoint| head -n 1| awk -F'"' '{print $2}' | awk -F'-' '{print $NF}'`
    model_name=output/model.ckpt-${model_step}
else
    BERT_BASE_DIR=xxxx/chinese_L-12_H-768_A-12/
    model_name=bert_model.ckpt
fi


hadoop fs -rmr $OUTPUT
    
hadoop streaming  \
    -input ${INPUT}/* \
    -output ${OUTPUT}\
    -cacheArchive ${HADOOP_PYTHON_PATH}#python-2.7.14 \
    -cacheFile ${BERT_BASE_DIR}/vocab.txt#vocab.txt \
    -cacheFile ${BERT_BASE_DIR}/bert_config.json#bert_config.json \
    -cacheFile ${BERT_BASE_DIR}/$model_name.data-00000-of-00001#bert_model.ckpt.data-00000-of-00001 \
    -cacheFile ${BERT_BASE_DIR}/$model_name.index#bert_model.ckpt.index \
    -cacheFile ${BERT_BASE_DIR}/$model_name.meta#bert_model.ckpt.meta \
    -files "./bin/,./scripts/" \
    -mapper "cat" \
    -reducer "sh -x scripts/extract_feature_bert_hadoop.sh" \
    -jobconf mapred.job.name=$JOB_NAME  \
    -jobconf mapred.job.map.capacity=500     \
    -jobconf mapred.job.reduce.capacity=2000        \
    -jobconf mapred.reduce.tasks=2000                \
    -jobconf stream.memory.limit=2000 \
    -jobconf mapred.job.queue.name=xxxx \
    -jobconf mapred.job.priority=${JOB_PRI}

[[ $? -ne 0 ]] && exit 1

rm -rf $output_file
hadoop fs -getmerge $OUTPUT $output_file
return $?
}

function main()
{
    run
    [[ $? -ne 0 ]] && exit 1
    return 0
}
main 
