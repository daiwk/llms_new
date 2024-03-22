## posterior data

export LD_LIBRARY_PATH=`pwd`/:$LD_LIBRARY_PATH
export WIDTH=224
export HEIGHT=224
export MODEL_TYPE=MobileNet
export DIM=1024

##export WIDTH=224
##export HEIGHT=224
##export MODEL_TYPE=DenseNet201
##export DIM=1920
###
##export WIDTH=224
##export HEIGHT=224
##export MODEL_TYPE=NASNetMobile
##export DIM=1056
###
##export WIDTH=299
##export HEIGHT=299
##export MODEL_TYPE=InceptionResNetV2
##export DIM=1536
###

export DIM=64
ckpt=0004

port=8008
netstat -nltp| grep $port | awk '{print $NF}'| awk -F'/' '{print $1}'| xargs kill -9
nohup ./python3.7.1_gcc82_pd2.0rc1_cpu/bin/python3 ./python3.7.1_gcc82_pd2.0rc1_cpu/bin/tensorboard --logdir=logs --port=$port --host `hostname` &


python=./python-2.7.14/bin/python
python3=./python3.7.1_gcc82_pd2.0rc1_cpu/bin/python3

function get_info()
{
    rm -rf imgs/* 
    
    cat nid.only_168.res| python trans.py | sort -k2,2 -nr | head -n 50000 > nid.only_168.res.format

    prefix=nid.only_168.res.format

    file_num=50
    lines=`wc -l ./$prefix | awk -F' ' '{print $1}'`
    per_file_line=`echo "scale=0;$lines/$file_num + 1"| bc` # 保留0位小数，保证有file_num个文件

    split ./$prefix -l ${per_file_line} -d output/${prefix}_

    python gen_cmd.py $prefix $file_num > cmds.$prefix.sh

    time sh -x cmds.$prefix.sh

}

function get_info_infer()
{
    rm -rf imgs_infer/* 
    
    time cat badcase.$infer_flag.txt | $python read_xxxx.py bad_case.res imgs_infer 
}

function get_img_vec() 
{
#    export https_proxy=http://xxx:aaa
#    export http_proxy=http://xxx:aaa
    
    time $python3 -u multimodal_infer.py ./imgs/ ./img.vec.$MODEL_TYPE
}

function get_img_vec_infer() 
{
#    export https_proxy=http://xxx:aaa
#    export http_proxy=http://xxx:aaa
    
    time $python3 -u multimodal_infer.py ./imgs_infer/ ./img_infer.vec.$MODEL_TYPE 
}

function build_annoy() 
{
    time $python -u build_annoy.py
}

function recall_annoy() 
{
    time $python -u recall_annoy.py  > multimodal.$ckpt.recall.res.batch.has_show.$MODEL_TYPE.html
}

function recall_annoy_infer() 
{
    time $python -u recall_annoy_infer.py  > multimodal.$ckpt.recall.res.infer.has_show.$MODEL_TYPE.$infer_flag.html
}


function run_batch()
{
    get_info
    get_img_vec
    build_annoy
    recall_annoy
}

function train()
{
    $python3 -u multimodal_train.py imgs
}

function run()
{
    export infer_flag=$1
    get_info_infer
    get_img_vec_infer
    recall_annoy_infer
    port=8881
    netstat -nltp| grep $port | awk '{print $NF}'| awk -F'/' '{print $1}'| xargs kill -9
    python -m CGIHTTPServer $port
}


train > log/$MODEL_TYPE.train.log 2>&1 
run_batch > log/$MODEL_TYPE.batch.log.$ckpt 2>&1
run eye > log/$MODEL_TYPE.eye.log.$ckpt 2>&1
run mouth > log/$MODEL_TYPE.mouth.log.$ckpt 2>&1
