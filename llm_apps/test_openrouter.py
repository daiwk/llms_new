# https://openrouter.ai/deepseek/deepseek-r1-0528:free/api
# https://openrouter.ai/docs/api-reference/limits
# model ID 带 :free
# 每分钟最多 20 次请求（限速）
# 每天可调用 50 次 free 模型请求（如果你未充值 ≥ 10 美元）
# 充值 ≥ 10 美元 后，日调用上限提升至 1000 次

from openai import OpenAI

openrouter_api_key = "sk-or-v1-CCC714c1d21b9084f77c0f7e09ebc77a056ed1cdd5e41b64133CCCa5f20019d187CCCc855"

openrouter_api_key = openrouter_api_key.replace("CCC", "")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=openrouter_api_key,
)

completion = client.chat.completions.create(
#  extra_headers={
#    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
#    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
#  },
  extra_body={},
  # model="deepseek/deepseek-r1-0528:free",
  model="moonshotai/kimi-k2:free",
  messages=[
    {
      "role": "user",
      "content": "2025年nba总冠军是谁"
    }
  ]
)
print(completion.choices[0].message.content)
