#!/bin/sh
set -x

source ~/.bashrc

JOB_PRI=HIGH

DATE=`date +"%Y%m%d"`
DATE=`cat ./dict/today`
yesterday=`cat ./dict/yesterday`
HOUR=*

export use_official=1

output_file=$1
tag=news

function run()
{
INPUT=xxxx/${tag}_title/${DATE}

OUTPUT=xxxx/news_annoy_miniapp/${DATE}/

JOB_NAME="xxxx_news_annoy_miniapp_$DATE"

HADOOP_PYTHON_PATH=xxxx/python-2.7.14.tar.gz


hadoop fs -rmr $OUTPUT

rm -rf dict_hadoop
mkdir -p dict_hadoop/dict/

cp ./dict/bert_miniapp.annoy ./dict_hadoop/dict/
cp ./dict/bert_miniapp_idx ./dict_hadoop/dict/
cp ./dict/stopwords ./dict_hadoop/dict/

cd dict_hadoop/dict
tar vczf ./dict.tar.gz ./*

cd -

dict_path=xxxx/dict/

hadoop fs -rmr $dict_path
hadoop fs -mkdir $dict_path
hadoop fs -put dict_hadoop/dict/dict.tar.gz $dict_path


if [[ ${use_official} -eq 0 ]];then
    vec_dim=64
else
    vec_dim=768
fi

hadoop streaming  \
    -input ${INPUT}/* \
    -output ${OUTPUT} \
    -cacheArchive ${HADOOP_PYTHON_PATH}#python-2.7.14 \
    -cacheArchive ${dict_path}/dict.tar.gz#dict \
    -files "./bin/,./scripts/," \
    -mapper "./python-2.7.14/python-2.7.14/bin/python ./bin/bert/news_get_nearest_miniapp_with_bert_hadoop.py $vec_dim" \
    -reducer "cat" \
    -jobconf mapred.job.name=$JOB_NAME  \
    -jobconf mapred.job.map.capacity=500     \
    -jobconf mapred.job.reduce.capacity=500        \
    -jobconf mapred.reduce.tasks=2000                \
    -jobconf stream.memory.limit=8000 \
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
