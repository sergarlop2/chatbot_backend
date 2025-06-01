from langchain_community.document_loaders import PyPDFLoader 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

#EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
#EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
#EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-mpnet-base-v2"
#EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
#EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v2-base-es" # buen calculo en español, requiere trust remote code
PERSIST_DIR = "docs/chroma/"
CHUNK_SIZE_PAGES = 5 # Number of pages per chunk
CHUNK_OVERLAP_PAGES = int(0.2 * CHUNK_SIZE_PAGES) # 20% overlap

# Load PDFs
loaders = [
    PyPDFLoader("docs/DigComSCCI_processed.pdf"),
    PyPDFLoader("docs/SCCIModulation_processed.pdf"),
    PyPDFLoader("docs/Unit3.SelectiveCh_processed.pdf"),
    PyPDFLoader("docs/ChannelSCCI_processed.pdf"),
    PyPDFLoader("docs/SCCIdiversity_processed.pdf")
    #PyPDFLoader("docs/07-Modulaciones y constelaciones-guion.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Group pages by source document
doc_groups = {}
for doc in docs:
    source = doc.metadata["source"]
    if source not in doc_groups:
        doc_groups[source] = []
    doc_groups[source].append(doc)

# Create chunks with overlap
split_docs = []
for source, pages in doc_groups.items():
    # Order pages and prepare for chunking
    sorted_pages = sorted(pages, key=lambda x: x.metadata["page"])
    total_pages = len(sorted_pages)
    start = 0
    
    while start < total_pages:
        end = start + CHUNK_SIZE_PAGES
        chunk_pages = sorted_pages[start:end]
        
        # Combine text and metadata
        combined_text = "\n\n".join([p.page_content for p in chunk_pages])
        metadata =  {
            "source": source,
            "start_page": chunk_pages[0].metadata["page"],
            "end_page": chunk_pages[-1].metadata["page"],
            "total_chunk_pages": len(chunk_pages)
        }
        
        split_docs.append(Document(
            page_content=combined_text,
            metadata=metadata
        ))
        
        start += (CHUNK_SIZE_PAGES - CHUNK_OVERLAP_PAGES)

# Add chunk-specific metadata
for idx, chunk in enumerate(split_docs):
    chunk.metadata.update({
        "chunk_id": idx,
        "chunk_length": len(chunk.page_content)
    })

# Create embeddings and persist in Chroma
embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True} # better for cosine similarity
)
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=PERSIST_DIR,
    collection_metadata={"hnsw:space": "cosine"}
)

print("Number of document splits:", len(split_docs))
print("Number of collections in DB:", vectordb._collection.count())
print(f"Example text: {split_docs[18]}")