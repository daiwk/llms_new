from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

llm = Ollama(model="llama2")

output_parser = StrOutputParser()

chain = prompt | llm  | output_parser

ss = chain.invoke({"input": "who are you?"})

print(ss)
