
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

from vllm import LLM, SamplingParams

def merge_lora_and_save(base_model_path, lora_path, output_path):
    base_model = AutoModel.from_pretrained(base_model_path)

    model = PeftModel.from_pretrained(base_model, lora_path)

    model = model.merge_and_unload()

    model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

local_ckpt_path = "./gte-Qwen2-1.5B-instruct"
local_ckpt_path = "./gte-Qwen2-1.5B-instruct"
ckpt_file = "user_noi2i_mrl_lora_gte-bs8-acc32-lora8-multistage-steps500-thres0.9"

merged_model_path = "./merged_model"

#merge_lora_and_save(local_ckpt_path, ckpt_file, merged_model_path)

tensor_parallel_size=1
pipeline_parallel_size=1
llm = LLM(merged_model_path, tensor_parallel_size=tensor_parallel_size, distributed_executor_backend="mp", pipeline_parallel_size=pipeline_parallel_size, task="embed")#,  

batch_prompts = ["一个性别为未知的未知年龄用户的视频兴趣是：蔬菜种植，黄瓜种植，龙牙百合，丝瓜种植，冬瓜，农村大席菜，王树国，香瓜，田地，按照时长兴趣强度由高到低点击了如下类别的直播：主播闲聊，美食制作展示，商品销售，山村野食，流行方向。", "bbb"]
batch_uids = [1, 2]

target_dim = 64
preds = llm.embed(batch_prompts)
# preds = model.encode(batch_prompts, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size, device="cuda")
# preds = self.llm.encode(batch_prompts, convert_to_numpy=True, normalize_embeddings=True, device="cuda")
xidx = 0
results = []
for output in preds:
    prompt = batch_prompts[xidx]
    pid = batch_uids[xidx]
    # embeds = output
    embeds = output.outputs.embedding[: target_dim]
    print(f"Prompt: {prompt!r}, Embeddings: {embeds!r}\n")
    # cur_emb = [round(v, 6) for v in embeds]
    # print(f"Prompt: {prompt!r}, short emb: {bottleneck!r}, recon emb: {reconstructed!r}\n")
    results.append(embeds)
    # ps.write xxx
    xidx += 1