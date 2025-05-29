 

import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import time

from vllm import LLM, SamplingParams

local_ckpt_path = "./gte-Qwen2-1.5B-instruct"
local_ckpt_path = "./gte-Qwen2-1.5B-instruct"
ckpt_file = "user_noi2i_mrl_lora_gte-bs8-acc32-lora8-multistage-steps500-thres0.9"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

merged_model_path = "./merged_model"

#merge_lora_and_save(local_ckpt_path, ckpt_file, merged_model_path)

target_dim = 64

batch_prompts = []
batch_uids = []
i = 0
with open("./x", "r") as fin:
#with open("./x.head", "r") as fin:
    for line in fin:
        line = line.strip("\n")
        batch_prompts.append(line)
        batch_uids.append(i)
        i += 1


def merge_lora_and_save(base_model_path, lora_path, output_path):
    base_model = AutoModel.from_pretrained(base_model_path)

    model = PeftModel.from_pretrained(base_model, lora_path)

    model = model.merge_and_unload()

    model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

def run():
    tensor_parallel_size=1
    pipeline_parallel_size=1
    #llm = LLM(merged_model_path, tensor_parallel_size=tensor_parallel_size, distributed_executor_backend="mp", pipeline_parallel_size=pipeline_parallel_size, task="embed")#,  
    llm = LLM(merged_model_path, tensor_parallel_size=tensor_parallel_size, distributed_executor_backend="mp", pipeline_parallel_size=pipeline_parallel_size, task="embed", dtype=torch.bfloat16)#,  
   
    model = SentenceTransformer(local_ckpt_path, truncate_dim=target_dim)#.to(dtype=torch.bfloat16, device="cuda")
    #model = torch.compile(model)
    model.load_adapter(ckpt_file)
    llm = model.to(dtype=torch.bfloat16, device="cuda")
    #llm = model.to(device="cuda")
    pool = llm.start_multi_process_pool()
    
    start = time.time()
    # preds = llm.embed(batch_prompts)
    preds = model.encode(batch_prompts, convert_to_numpy=True, normalize_embeddings=False,device="cuda")
    print_res(preds)
    end = time.time()
    print(end-start)
    
    print("xxx"*10)

    start = time.time()
    preds = llm.encode_multi_process(batch_prompts, pool)#, batch_size=256)
    print_res(preds)
    end = time.time()
    print(end-start)
    llm.stop_multi_process_pool(pool)
    
def print_res(preds):
    # preds = self.llm.encode(batch_prompts, convert_to_numpy=True, normalize_embeddings=True, device="cuda")
    xidx = 0
    results = []
    for output in preds:
        prompt = batch_prompts[xidx]
        pid = batch_uids[xidx]
        embeds = output
        # embeds = output.outputs.embedding[: target_dim]
        #print(f"Prompt: {prompt!r}, Embeddings: {embeds!r}\n")
        # cur_emb = [round(v, 6) for v in embeds]
        # print(f"Prompt: {prompt!r}, short emb: {bottleneck!r}, recon emb: {reconstructed!r}\n")
        results.append(embeds)
        # ps.write xxx
        xidx += 1
if __name__ == "__main__":
    run()