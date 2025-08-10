from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Updated import

def get_embedding_function():
    """Returns the HuggingFace embeddings for ChromaDB."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
