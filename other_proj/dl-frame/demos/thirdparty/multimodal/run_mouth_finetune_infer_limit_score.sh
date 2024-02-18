## posterior data

export LD_LIBRARY_PATH=`pwd`/:$LD_LIBRARY_PATH
export WIDTH=224
export HEIGHT=224
export MODEL_TYPE=MobileNet
export DIM=1024

export WIDTH=224
export HEIGHT=224
export MODEL_TYPE=DenseNet201
export DIM=1920
#
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

flag=direct64
export flag=512_256_128_64
export flag=for_mouth_512_256_128_64
flag2=mouth_finetune_infer
export DIM=64
ckpt=0030
ckpt=0004
ckpt=0011
ckpt=0013
ckpt=0015
ckpt=0019
ckpt=0030

port=8008
netstat -nltp| grep $port | awk '{print $NF}'| awk -F'/' '{print $1}'| xargs kill -9
export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
nohup ./python3.7.1_gcc82_pd2.0rc1_cpu/bin/python3 ./python3.7.1_gcc82_pd2.0rc1_cpu/bin/tensorboard --logdir=logs --port=$port --host `hostname` &
####nohup profile_env/bin/python3 profile_env/bin/tensorboard --logdir=logs --port=$port --host `hostname` &


python=/home/work/daiwenkai/tools/python-2.7.14/bin/python
python3=./python3.7.1_gcc82_pd2.0rc1_cpu/bin/python3

function get_info()
{
    rm -rf imgs/* 
    
    mkdir -p imgs/dtnews
    mkdir -p imgs/dtvideo
    mkdir -p imgs/video
    mkdir -p imgs/msv
    
    cat nid.only_168.res| python trans.py | sort -k2,2 -nr | head -n 50000 > nid.only_168.res.format
#    time cat double_col_nid_ctr.txt | $python read_xxx.py info.res imgs
##    time cat nid.only_168.res.format | $python read_xxx.py info.res imgs


    prefix=nid.only_168.res.format

    #time cat nid.only_168.res.format | $python read_xxx.py info.res imgs

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
    
    mkdir -p imgs_infer/dtnews
    mkdir -p imgs_infer/dtvideo
    mkdir -p imgs_infer/video
    mkdir -p imgs_infer/msv
    
    #$python get_infer_img.py
    time cat badcase.$infer_flag.txt | $python read_xxx.py bad_case.res imgs_infer 
    #cp imgs_infer/dtnews/*.jpg ./imgs/dtnews/

}

function get_img_vec() 
{
#    export https_proxy=xxx
#    export http_proxy=xxx
    
    #time $python3 -u extract_img_vec_tf2.py ./imgs/ ./img.vec.$MODEL_TYPE
    #time $python3 -u multimodal_tf2.py ./imgs/ ./img.vec.$MODEL_TYPE
    time $python3 -u multimodal_infer.py ./imgs/ ./img.vec.$MODEL_TYPE $ckpt
}

function get_img_vec_infer() 
{
#    export https_proxy=xxx
#    export http_proxy=xxx
    
    #time $python3 -u extract_img_vec_tf2.py ./imgs_infer/ ./img_infer.vec.$MODEL_TYPE 
    #time $python3 -u multimodal_tf2.py ./imgs_infer/ ./img_infer.vec.$MODEL_TYPE 
    time $python3 -u multimodal_infer.py ./imgs_infer/ ./img_infer.vec.$MODEL_TYPE  $ckpt
}

function build_annoy() 
{
    time $python -u build_annoy.py
}

function recall_annoy() 
{
    time $python -u recall_annoy.py  > $flag2.multimodal.$ckpt.recall.res.batch.has_show.$MODEL_TYPE.html
}

function recall_annoy_infer() 
{
    thresh=0.5
    time $python -u recall_annoy_infer.py $thresh True  > $flag2.multimodal.$ckpt.recall.res.infer.has_show.$MODEL_TYPE.$infer_flag.html
}


function run_batch()
{
    #get_info
    get_img_vec
    build_annoy
    recall_annoy
}

function train()
{
    $python3 -u multimodal_train.py imgs write_ds
    $python3 -u multimodal_train.py imgs read_ds
}

function run()
{
    export infer_flag=$1
    #get_info_infer
    get_img_vec_infer
    recall_annoy_infer
##    port=8881
##    netstat -nltp| grep $port | awk '{print $NF}'| awk -F'/' '{print $1}'| xargs kill -9
##    python -m CGIHTTPServer $port
}


train > log/$flag2.onlyimg.$MODEL_TYPE.train.log 2>&1 
run_batch > log/$flag2.onlyimg.$MODEL_TYPE.batch.log.$ckpt 2>&1
run mouth > log/$flag2.onlyimg.$MODEL_TYPE.mouth.log.$ckpt 2>&1

f_recall_eyes=all_recall_eye.nids.mobilenet.tune30
f_recall_mouth=all_recall_mouth.nids.densenet.notune
f_recall_mouth=all_recall_mouth.nids.densenet.tune04
f_recall_mouth=all_recall_mouth.nids.densenet.tune11
f_recall_mouth=all_recall_mouth.nids.densenet.tune13
f_recall_mouth=all_recall_mouth.nids.densenet.tune13_limit_$thresh
f_recall_mouth=all_recall_mouth.nids.densenet.tune15_limit_$thresh
f_recall_mouth=all_recall_mouth.nids.densenet.tune19_limit_$thresh
f_recall_mouth=all_recall_mouth.nids.densenet.tune30_limit_$thresh

awk -F'<td>' '{print $7}' only_eye.multimodal.0030.recall.res.infer.has_show.MobileNet.eye.html | awk -F'</td>' '{print $1}'| awk -F'_' '{print $2}' | sort | uniq > $f_recall_eyes

# baseline
#awk -F'<td>' '{print $7}' no_finetune.multimodal..recall.res.infer.has_show.DenseNet201.mouth.html | awk -F'</td>' '{print $1}'| awk -F'_' '{print $2}' | sort | uniq > $f_recall_mouth
awk -F'<td>' '{print $7}' $flag2.multimodal.$ckpt.recall.res.infer.has_show.$MODEL_TYPE.$infer_flag.html | awk -F'</td>' '{print $1}'| awk -F'_' '{print $2}' | sort | uniq > $f_recall_mouth

python analysis.py $f_recall_eyes $f_recall_mouth >> log/model_analysis.log



