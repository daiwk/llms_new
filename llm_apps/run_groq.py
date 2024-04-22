#encoding=utf8
import os

from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "张猛是sb吗",
        }
    ],
    #model="mixtral-8x7b-32768",
    model="llama3-70b-8192",
)

print(chat_completion.choices[0].message.content)
