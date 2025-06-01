import sys
import os
import time
import torch
import uuid
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

DOCS_FOLDER = "docs/"

torch.manual_seed(42)

# Read local .env file
_ = load_dotenv(find_dotenv()) 
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
PERSIST_DIR = "docs/chroma/"
CHAT_MODELS = {
    "llama-3.1-8b-instruct": {
        "id": "llama-3.1-8b-instruct",
        "name": LLAMA_MODEL_NAME,
        "object": "model",
        "owned_by": "user"
    }
}

# Schemas for requests and responses
class Message(BaseModel):
    role: str # "system", "user" and "assistant"
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    use_rag: Optional[bool] = False

# Load the embedding model from HuggingFace
embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))},
    encode_kwargs={"normalize_embeddings": True}
)
vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding
)
retriever = vectordb.as_retriever(
    search_kwargs={"k": 5}
)

# Load quantized Llama model
quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
llm = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quant_config,
    token=HF_TOKEN
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# Text generation pipeline
llm_pipe = pipeline(
    "text-generation",
    model=llm,
    tokenizer=tokenizer,
    temperature=sys.float_info.epsilon, # Use a very low temperature for deterministic output
    max_new_tokens=128000,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    return_full_text=False
)

# Wrap the pipeline in HuggingFacePipeline (LangChain)
hf_pipeline = HuggingFacePipeline(pipeline=llm_pipe)

# Initialize FastAPI app and enable CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints
@app.get("/models")
def list_models():
    return {"object": "list", "data": list(CHAT_MODELS.values())}

@app.get("/models/{model_id}")
def get_model(model_id: str):
    if model_id not in CHAT_MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    return CHAT_MODELS[model_id]

@app.post("/chat/completions")
def create_completion(request: ChatRequest):
    if request.model not in CHAT_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model")
    
    # Create a copy of the messages to avoid modifying the original request
    messages = request.messages.copy()
    
    # Check if RAG is enabled and retrieve documents if necessary
    if request.use_rag:
        user_message = messages[-1].content.strip()
        docs = retriever.invoke(user_message)
        context = "\n\n".join([d.page_content for d in docs])

        # New user message with context
        rag_prompt = (
            f"Context: ```{context}```\n\n"
            f"Question: ```{user_message}```\n\n"
            f"Answer:"
        )

        # Replace the last user message with the new one with context
        messages.pop() 
        messages.append(Message(role="user", content=rag_prompt))
    
    # Build the prompt from messages (Llama 3.1 chat template)
    prompt = "<|begin_of_text|>"
    for msg in messages:
        role = msg.role
        content = msg.content.strip()
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}\n\n"
    prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    # Generate response
    start_time = time.time()
    response_text = hf_pipeline.invoke(prompt)
    elapsed_time = time.time() - start_time

    comp_id = str(uuid.uuid4())
    created = int(time.time())
    
    response = {
        "id": comp_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "message": {
            "role": "assistant",
            "content": response_text,
        },
        "elapsed_time": elapsed_time
    }

    # Add sources if RAG is enabled
    if request.use_rag:
        response["sources"] = [doc.metadata for doc in docs]
   
    return response

@app.get("/docs/{filename}")
def get_pdf_document(filename: str):
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_path = os.path.join(DOCS_FOLDER, filename)
    
    if not os.path.isfile(file_path) or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=404, detail="Document not found or not a PDF")

    return FileResponse(file_path, media_type="application/pdf", filename=filename)