from transformers import RecurrentGemmaForCausalLM, RecurrentGemmaConfig
import torch
import time

def init_model(local_name):
    # Initializing a RecurrentGemma recurrentgemma-2b style configuration
    configuration = RecurrentGemmaConfig()
    configuration.intermediate_size = 15360
    configuration.num_key_value_heads = 1
    configuration.embeddings_scale_by_sqrt_dim = True
    
    # Initializing a model from the recurrentgemma-2b style configuration
    model = RecurrentGemmaForCausalLM(configuration).cuda() #.to(torch.bfloat16) ## because torch.triu dosen't supply bf16
    
    # Accessing the model configuration
    configuration = model.config
    model.save_pretrained(local_name)

from transformers import AutoTokenizer

local_name = "recurrent_gemma_dwk_empty"
init_model(local_name)
model = RecurrentGemmaForCausalLM.from_pretrained(local_name).cuda()
tokenizer = AutoTokenizer.from_pretrained("./tokenizer_recurrent_gemma_dwk/")

instances = []
bs = 32
with open('input.txt', 'r') as f:
    tmp_batch = []
    for line in f:
        line = line.strip("\n")
        idx = 0
        if idx > 0 and idx % bs == 0:
            tmp_batch = []
            inputs = tokenizer(tmp_batch, return_tensors='pt')
            inputs = inputs.to("cuda")
            instances.append(inputs)
        else:
            chat = [
                        { "role": "user", "content": line},
                        ]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            print(prompt)
            tmp_batch.append(prompt)
            idx += 1

    ## last batch
    inputs = tokenizer(tmp_batch, return_tensors='pt')
    inputs = inputs.to("cuda")
    instances.append(inputs)

t1 = time.time()
for inputs in instances:
    generate_ids = model.generate(**inputs, max_new_tokens=150, repetition_penalty=1.1)
    res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(res)
t2 = time.time()
print("time:", (t2 - t1) *1.0 / len(instances))

