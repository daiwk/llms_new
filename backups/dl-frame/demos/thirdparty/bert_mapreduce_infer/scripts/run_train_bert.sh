
source ~/.bashrc

DATE=`date +"%Y%m%d"`
export BERT_BASE_DIR=./bert_model/chinese_wordpiece_small/
export HADOOP_BERT_BASE_DIR=xxxx/b_model/

input_file=./dict/sample.txt
tfrecord_file=./dict/sample.tfrecord
file_list=""
output_path=$BERT_BASE_DIR/output_${param_name}/

function prepare_train_data()
{
    cat ./dict/news_title dict/miniapp_title > ./dict/all_title
    cat ./dict/all_title | ./python-2.7.14/bin/python ./bin/bert/prepare_train_data.py > $input_file
    sh -x scripts/split_file.sh $input_file 200000 dict/file_indices.all_title
    [[ $? -ne 0 ]] && exit 1
    return 0
}

function create_pretraining_data()
{

    for idx in `cat dict/file_indices.all_title`
    do
    {
        ./python-2.7.14/bin/python ./bin/bert/create_pretraining_data.py \
            --input_file=$input_file$idx \
            --output_file=$tfrecord_file.$idx.${param_name} \
            --vocab_file=$BERT_BASE_DIR/vocab.txt \
            --do_lower_case=True \
            --max_seq_length=${g_max_seq_length} \
            --max_predictions_per_seq=${g_max_predictions_per_seq} \
            --masked_lm_prob=${g_masked_lm_prob} \
            --random_seed=12345 \
            --dupe_factor=${g_dupe_factor}
    } &
    done
    wait
    [[ $? -ne 0 ]] && exit 1


    return 0
}

function run_pretraining_offline()
{
    file_list=""
    for idx in `cat dict/file_indices.all_title`
    do
        file_list=$tfrecord_file.$idx.${param_name},$file_list
    done

    ./python-2.7.14/bin/python ./bin/bert/run_pretraining.py \
        --input_file=$file_list \
        --output_dir=$output_path \
        --do_train=True \
        --do_eval=True \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --train_batch_size=${g_train_batch_size} \
        --max_seq_length=${g_max_seq_length} \
        --max_predictions_per_seq=${g_max_predictions_per_seq} \
        --num_train_steps=${g_num_train_steps} \
        --num_warmup_steps=10 \
        --learning_rate=1e-4
    [[ $? -ne 0 ]] && exit 1
    return 0
}

function run_pretraining_online()
{
    file_list=""
    for idx in `cat dict/file_indices.all_title`
    do
        file_list=$tfrecord_file.$idx,$file_list
    done

    ./python-2.7.14/bin/python ./bin/bert/run_pretraining.py \
        --input_file=$file_list \
        --output_dir=$output_path \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --do_train=True \
        --do_eval=True \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --train_batch_size=${g_train_batch_size} \
        --max_seq_length=${g_max_seq_length} \
        --max_predictions_per_seq=${g_max_predictions_per_seq} \
        --num_train_steps=${g_num_train_steps} \
        --num_warmup_steps=10 \
        --learning_rate=2e-5
    [[ $? -ne 0 ]] && exit 1
    return 0
}

function upload_model()
{
    hadoop fs -rmr $HADOOP_BERT_BASE_DIR/$DATE
    hadoop fs -mkdir $HADOOP_BERT_BASE_DIR/$DATE
    hadoop fs -put $BERT_BASE_DIR $HADOOP_BERT_BASE_DIR/$DATE

}

function main()
{
    prepare_train_data
    [[ $? -ne 0 ]] && exit 1
    create_pretraining_data
    [[ $? -ne 0 ]] && exit 1

    if [[ x$train_mode == x"online" ]];then
        run_pretraining_online
        [[ $? -ne 0 ]] && exit 1
    elif [[ x$train_mode == x"offline" ]];then
        run_pretraining_offline
        [[ $? -ne 0 ]] && exit 1
    fi
    upload_model
    [[ $? -ne 0 ]] && exit 1
    return 0
}

main 2>&1 
