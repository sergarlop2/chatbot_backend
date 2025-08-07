from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv, find_dotenv
import os
import torch

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

#_ = load_dotenv(find_dotenv()) 
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))},
    encode_kwargs={"normalize_embeddings": True}
)

llm = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN
)

tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME, token=HF_TOKEN)