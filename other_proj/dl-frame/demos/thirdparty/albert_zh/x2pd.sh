####/home/work/tools/python-3-tf-1.14.0-gpu/bin/x2paddle -f tensorflow -m ./albert_dwk.pb -s pd-model --without_data_format_optimization
#/home/work/tools/python-3-tf-1.14.0-gpu/bin/x2paddle -f tensorflow -m ./albert_dwk_origin.pb -s pd-model-origin --without_data_format_optimization

/home/work/tools/python-3-tf-1.14.0-gpu/bin/python3.6 tf2pd.py

## 有dropout
/home/work/tools/python-3-tf-1.14.0-gpu/bin/x2paddle -f tensorflow -m ./albert_dwk.pb -s pd-model-merge --without_data_format_optimization --params_merge

cd raw-paddle/model_with_code
/home/work/tools/python-3-tf-1.14.0-gpu/bin/python3.6 model.py
/home/work/tools/python-3-tf-1.14.0-gpu/bin/python3.6 model_batchsize.py

cd -

/home/work/tools/python-3-tf-1.14.0-gpu/bin/python3.6 gen_tokens.py > tokens.log
/home/work/tools/python-3-tf-1.14.0-gpu/bin/python3.6 fluid_infer_merge.py > fluid.res
