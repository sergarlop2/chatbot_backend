import os
import sys
import argparse
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Chat with Llama 3.1-8B-Instruct in the terminal.")
parser.add_argument("--rag", action="store_true", help="Enable RAG (Retrieval-Augmented Generation).")
parser.add_argument("--memory", action="store_true", help="Enable chat memory (keep conversation history).")
args = parser.parse_args()

# Configuration
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
PERSIST_DIR = "docs/chroma/"

torch.manual_seed(42)

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
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

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
    temperature=sys.float_info.epsilon,  # Use a very low temperature for deterministic output
    max_new_tokens=128000,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    return_full_text=False
)

# Wrap the pipeline in HuggingFacePipeline (LangChain)
hf_pipeline = HuggingFacePipeline(pipeline=llm_pipe)

# System prompt
system_prompt = "You are an AI assistant expert in digital communications"

# Chat history (only used if memory is enabled)
messages = []
messages.append({
    "role": "system",
    "content": system_prompt
})

print("\nChat with Llama 3.1-8B-Instruct (type 'exit' to quit)\n")
print(f"RAG: {'ON' if args.rag else 'OFF'} | Memory: {'ON' if args.memory else 'OFF'}")

while True:
    user_input = input("\nUser: ").strip()
    if user_input.lower() in ["exit", "quit", "q"]:
        print("\nExiting chat..")
        break

    # If RAG is enabled, retrieve context from the vector database
    if args.rag:
        docs = retriever.invoke(user_input)
        context = "\n\n".join([d.page_content for d in docs])
        user_message = f"Context: ```{context}```\n\nQuestion: ```{user_input}```\n\nAnswer:"
    else:
        user_message = user_input

    if args.memory:
        # Add message to chat history
        messages.append({"role": "user", "content": user_message})

        # Build the prompt using messages from history (Llama 3.1 chat template)
        prompt = "<|begin_of_text|>"
        for msg in messages:
            prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}\n\n"
        prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        # Build a prompt only with the last user message
        prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" \
                 f"{system_prompt}\n\n" \
                 f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}\n\n" \
                 "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    # Generate the response
    response = hf_pipeline.invoke(prompt)

    print(f"\nAssistant: {response}")

    if args.memory:
        messages.append({"role": "assistant", "content": response})
