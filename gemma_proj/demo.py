
import torch
from transformers import AutoTokenizer, GemmaForCausalLM

model = GemmaForCausalLM.from_pretrained("google/gemma-2b-it", torch_dtype=torch.bfloat16)
model.save_pretrained("gemma-2b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

prompt = "google is a "
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=3000)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

