## refs: https://github.com/leptonai/search_with_lepton/blob/main/README.md
#RELATED_QUESTIONS=1 LLM_MODEL=aaa BING_SEARCH_V7_SUBSCRIPTION_KEY=123 BACKEND=BING python rag.py

export OPENAI_API_KEY='sk-xxxx'
export GOOGLE_SEARCH_API_KEY=xxx
source ../token.sh
echo "-----------gpt3.5-----------"
LLM_MODEL=gpt-3.5-turbo-1106 GOOGLE_SEARCH_CX=14c612a1e778d40b3 BACKEND=GOOGLE python rag.py
echo "-----------gpt4-------------"
LLM_MODEL=gpt-4 GOOGLE_SEARCH_CX=14c612a1e778d40b3 BACKEND=GOOGLE python rag.py
