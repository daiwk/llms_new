from dataclasses import dataclass, field
import os
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from datasets import Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer

import pandas as pd


tqdm.pandas()

#from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#quantization_config = BitsAndBytesConfig(load_in_8bit=True)

def save_model(model_name, local_name):

#    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.save_pretrained(local_name + "_bf16")
#    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
#    model.save_pretrained(local_name + "_fp32")
    #model = AutoModel.from_pretrained(model_name, quantization_config=quantization_config) ## cuda only
    #model.save_pretrained(local_name + "_int8")
    if "OpenELM" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("./Meta-Llama-3-8B-Instruct", trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(local_name + "_tokenizer")
    os.system("cp ./%s_tokenizer/* ./%s_bf16" % (local_name, local_name))

def save_ds(name, local_name):
    dataset = load_dataset(name, split="train")
    dataset.to_csv(local_name)


model_map ={
	"facebook/opt-125m": "./opt-125m",
#	"Qwen/Qwen2.5-7B-Instruct": "./Qwen2.5-7B-Instruct",
#     "THUDM/chatglm3-6b": "./chatglm3-6b",
#     "./Meta-Llama-3-8B-Instruct": "./Meta-Llama-3-8B-Instruct",
#     "./Mistral-7B-Instruct-v0.2": "./Mistral-7B-Instruct-v0.2",
#     "./OpenELM-3B-Instruct": "./OpenELM-3B-Instruct",
#     "./Phi-3-mini-128k-instruct": "./Phi-3-mini-128k-instruct",
#     "./gemma-1.1-7b-it": "./gemma-1.1-7b-it",
#     "./recurrentgemma-2b-it": "./recurrentgemma-2b-it",
#     "./Qwen1.5-7B-Chat-GPTQ-Int8": "./Qwen1.5-7B-Chat-GPTQ-Int8",
#     "./Qwen1.5-7B-Chat": "./Qwen1.5-7B-Chat",
#     "./Octopus-v4": "./Octopus-v4",
}

for k in model_map:
    save_model(k, model_map[k])

