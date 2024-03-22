export LD_LIBRARY_PATH=~/cuda-9.0/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/cudnn/cudnn_v7.1/cuda/lib64/:$LD_LIBRARY_PATH

/opt/compiler/gcc-4.8.2/lib/ld-linux-x86-64.so.2 --library-path /opt/compiler/gcc-4.8.2/lib ~/tools/python-3-tf-2.0-gpu/bin/python3.6  ./dlrm_s_pytorch.py 
