from transformers import AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer
import time
import os
import torch
#model_name = "./baichuan2-7b-sft_baseline"
#model_name = "./chatglm3-6b_bf16"
model_name = "./baichuan2-7b-sft_prompt1_step5k_nolora"

#quantization_config = BitsAndBytesConfig(load_in_8bit=True)
#quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

def save(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    #model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, trust_remote_code=True) ## cuda only
    model = model.quantize(8).cuda() 
    fname = model_name + "_int8"
    model.save_pretrained(fname)

def use_int8(model_name):
    fname = model_name + "_int8"
    #model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cuda()
    model1 = model#.cuda() 
    #model1 = torch.nn.DataParallel(model1)
    model3 = model.quantize(8).cuda() 
    #model3 = torch.nn.DataParallel(model3)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    with open('input.txt', 'r') as f:
        for line in f:
            line = line.strip("\n")
            bs = 32
            iii = []
            for j in range(bs):
                aa = []
                for i in range(bs):
                    aa.append(line)
                inputs = tokenizer(aa, return_tensors='pt')
                inputs = inputs.to("cuda")
                iii.append(inputs)

        t1 = time.time()
        for inputs in iii:
            pred = model1.generate(**inputs, max_new_tokens=150, repetition_penalty=1.1)
            #print("bf16:", tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
        t2 = time.time()
        print("bf16:", (t2 - t1) *1.0 / len(iii))
    
        t1 = time.time()
        for inputs in iii:
            pred = model3.generate(**inputs, max_new_tokens=150, repetition_penalty=1.1)
            #print("int8_fix:", tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
        t2 = time.time()
        print("int8:", (t2 - t1) *1.0 / len(iii))
#save(model_name)
use_int8(model_name)
