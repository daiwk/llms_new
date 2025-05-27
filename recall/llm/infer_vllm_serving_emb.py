import subprocess
import time
import requests
import json
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer

import openai

target_dim = 64

def merge_lora_and_save(base_model_path, lora_path, output_path):
    base_model = AutoModel.from_pretrained(base_model_path)

    model = PeftModel.from_pretrained(base_model, lora_path)

    model = model.merge_and_unload()

    model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)


local_ckpt_path = "./gte-Qwen2-1.5B-instruct"
local_ckpt_path = "./local_model"
ckpt_file = "user_noi2i_mrl_lora_gte-bs8-acc32-lora8-multistage-steps500-thres0.9"

merged_model_path = "./merged_model"

merge_lora_and_save(local_ckpt_path, ckpt_file, merged_model_path)

model = merged_model_path

# model="/xx/data/model/code/torch_model/DeepSeek-R1-Distill-Qwen-32B-AWQ"
# tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

host = "localhost"
port =12345 


# 检查端口是否可用
def is_port_open(host, port, timeout=5):
    """
    检查指定端口是否可用
    :param host: 主机地址
    :param port: 端口号
    :param timeout: 超时时间（秒）
    :return: True 如果端口可用，否则 False
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError):
        return False

# 启动 vLLM 服务
def start_vllm_service_emb():
#        "CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
    port = 12345
    import os
    import tempfile

    # 创建临时目录并设置权限
    temp_dir = tempfile.mkdtemp()
    os.chmod(temp_dir, 0o755)
    
    # 设置环境变量
    env = os.environ.copy()
    env.update({
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        #"CUDA_VISIBLE_DEVICES": "0",
        "TMPDIR": temp_dir,
        "TEMP": temp_dir,
        "TMP": temp_dir
    })
    cur_port = port    
    tp_size = 1
    process = subprocess.Popen(

                ["python3", "-m", "vllm.entrypoints.openai.api_server",
                "--host", "0.0.0.0",
                "--port", str(cur_port),
                "--max-model-len", "1700",
                "--max-num-batched-tokens", "4096",
                #"--disable-log-requests",
                "--uvicorn-log-level", "warning",
                # "--max-num-seqs", "64",
                "--max-num-seqs", "128",
                # "--swap-space", "16",
                "--trust-remote-code",
                # "--enable-reasoning",
                #"--chat-template", "./less3.jinja",
                "--enable-prefix-caching",
                #"--enable-chunked-prefill",
                # "--enforce-eager",
                # "--reasoning-parser", "deepseek_r1",
                "--tensor-parallel-size", str(tp_size),
                "--served-model-name", "deepseek-reasoner",
        "--model", merged_model_path, 

    ],
        #shell=True,
        env=env,
        stdout=open("./out.log", "w"),
        stderr=open("./err.log", "w")
    )
    return process

# 发送请求的函数
def send_request_emb(prompt):
    """
    向 vLLM 服务发送请求
    :param prompt: 用户输入的提示
    :return: 模型生成的响应
    """
    client = openai.OpenAI(
        base_url=f"http://{host}:{port}/v1", # "http://<Your api-server IP>:port"
        api_key = "sk-no-key-required"
    )

    models = client.models.list()
    model = models.data[0].id

    try:
        messages = [{"role": "user", "content": prompt}]
        responses = client.embeddings.create(
            input=[
                prompt,
            ],
            model=model,
        )
    
        contents = []
        for data in responses.data:
            contents.append(data.embedding[:target_dim]) 

        return messages, contents
    except Exception as e:
        return {"error": str(e)}

# 主函数
def main():
    """
    主函数：启动服务、发送并发请求、处理结果
    """
    # 启动 vLLM 服务
    print("Starting vLLM service...")
    vllm_process = start_vllm_service_emb()

    # 检查端口是否可用
    print(f"Waiting for service to start on {host}:{port}...")
    while not is_port_open(host, port):
        print("Service not ready yet, waiting...")
        time.sleep(2)  # 每 2 秒检查一次
    print("Service is ready!")

    all_p = []
    with open("./x", 'r') as fin:
    #with open("./prompts", 'r') as fin:
        for line in fin:
            prompt = line.strip("\n")
            all_p.append(prompt)


    prompts = all_p[:10]


    # 使用 ThreadPoolExecutor 创建线程池并发发送请求
    print("Sending requests concurrently...")
    start = time.time()

    preds = []
    with ThreadPoolExecutor(max_workers=128) as executor:
        raw_preds = list(executor.map(send_request_emb, prompts))
#        for x in preds:
#            print(x)

        for x in raw_preds:
            prompt = x[0]
            #reasoning_content = x[1]
            content = x[1]
            #if random.randint(0,99) < 3:
            if True:
                # print(f'prompt: {prompt[0]["content"]!r} reasoning: {reasoning_content!r} res: {content!r}')
                print(f'prompt: {prompt[0]["content"]!r} res: {content!r}')
            preds.append(content)
        
    end = time.time()
    cost = end - start
    print(cost)
    # 关闭 vLLM 服务
    print("Stopping vLLM service...")
    vllm_process.terminate()

if __name__ == "__main__":
    main()