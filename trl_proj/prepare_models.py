
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


def save_model(model_name, local_name):
    
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(local_name)
    tokenizer.save_pretrained(local_name + "_tokenizer")

def save_ds(name, local_name):
    dataset = load_dataset(name, split="train")
    dataset.to_csv(local_name)


model_map ={
    "facebook/opt-350m": "./std/models/opt-350m", 
    "lvwerra/gpt2-imdb": "./std/models/gpt2-imdb", 
    "lvwerra/distilbert-imdb": "./std/models/distilbert-imdb"
    
}

for k in model_map:
    save_model(k, model_map[k])


ds_map ={
    "timdettmers/openassistant-guanaco": "std/datasets/openassistant-guanaco",
    "imdb": "std/datasets/imdb",
    "lucasmccabe-lmi/CodeAlpaca-20k": "std/datasets/CodeAlpaca-20k"
}
for k in ds_map:
    save_ds(k, ds_map[k])

