import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

GEMINI_API_KEY = os.environ['GOOGLE_AI_API_KEY']

llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

# model_name = 'qwen2:1.5b-instruct'
# llm = Ollama(model=model_name, base_url='https://68e3-103-241-244-6.ngrok-free.app')

# emb_model_name = "jinaai/jina-embeddings-v2-base-en"
# emb_model_kwargs = {'device': 'cpu'}
# emb_encode_kwargs = {'normalize_embeddings': False}

# embeddings_model = HuggingFaceEmbeddings(
#     model_name=emb_model_name,
#     model_kwargs=emb_model_kwargs,
#     encode_kwargs=emb_encode_kwargs
# )

def get_ai_resp(prompt):
    response = llm.invoke(prompt)

    print(response)

    return response


def get_ai_stream_resp(query):
    response = ""
    for chunks in llm.stream(query):
        print(chunks, end="")
        response += chunks
    
    return response
