# Embedding support for text, images, audio
from sentence_transformers import SentenceTransformer

def generate_text_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    return embeddings