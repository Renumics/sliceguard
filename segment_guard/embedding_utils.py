# Embedding support for text, images, audio
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import ViTFeatureExtractor, ViTModel
from tqdm import tqdm
from PIL import Image
import numpy as np

def generate_text_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    return embeddings



def generate_image_embeddings(image_paths, model_name="google/vit-base-patch16-224"):
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    model = ViTModel.from_pretrained(
        "google/vit-base-patch16-224", output_hidden_states=True
    )
    embeddings = []
    for i, image_path in enumerate(tqdm(image_paths)):
        try:
            with open(image_path, "rb") as f:
                image = Image.open(f)
                inputs = feature_extractor(images=image.convert("RGB"), return_tensors="pt")
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[0,0].cpu().detach().numpy()
                embeddings.append(emb)
                image.close()
        except:
            embeddings.append(None)
            print(f"Could not generate embedding for {image_path}.")
    embeddings = np.array([emb.tolist() if emb is not None else None for emb in embeddings])
    return embeddings
