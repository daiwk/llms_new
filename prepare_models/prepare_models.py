
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from datasets import Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer

from trl import SFTTrainer, is_xpu_available
import pandas as pd


tqdm.pandas()

#from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#quantization_config = BitsAndBytesConfig(load_in_8bit=True)


def save_model(model_name, local_name):
    
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.save_pretrained(local_name + "_bf16")
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(local_name + "_fp32")
    #model = AutoModel.from_pretrained(model_name, quantization_config=quantization_config) ## cuda only
    #model.save_pretrained(local_name + "_int8")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(local_name + "_tokenizer")

def save_ds(name, local_name):
    dataset = load_dataset(name, split="train")
    dataset.to_csv(local_name)


model_map ={
#    "facebook/opt-350m": "./opt-350m", 
#    "lvwerra/gpt2-imdb": "./gpt2-imdb", 
#    "lvwerra/distilbert-imdb": "./istilbert-imdb"
    "google/gemma-2b-it": "./gemma-2b-it",
    "google/gemma-7b-it": "./gemma-7b-it"

    
}

for k in model_map:
    save_model(k, model_map[k])


#ds_map ={
#    "timdettmers/openassistant-guanaco": "std/datasets/openassistant-guanaco",
#    "imdb": "std/datasets/imdb",
#    "lucasmccabe-lmi/CodeAlpaca-20k": "std/datasets/CodeAlpaca-20k"
#}
#for k in ds_map:
#    save_ds(k, ds_map[k])
#
