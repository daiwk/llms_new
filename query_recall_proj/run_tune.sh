date=20231231
xdir=/all_query_gip_cates/date=$date/
hadoop fs -cat $xdir/* > ./tune_data.txt.raw

export model_name=m3e-base
export new_model_name=tuned_m3e-base_${xtag}
pip3 install -r requirements.txt

hadoop fs -get /Latest_models/$model_name ./

python3 tune_m3e_hf.py ./tune_data.txt.raw 

#devices=0,1,2,3
#nproc_per_node=4
#CUDA_VISIBLE_DEVICES=$devices \
#    python3 -m torch.distributed.launch --nproc_per_node $nproc_per_node --master_port 8013 \
#    tune_m3e_hf_para.py 


hadoop fs -rmr /Latest_models/$new_model_name 
hadoop fs -put $new_model_name /Latest_models/
