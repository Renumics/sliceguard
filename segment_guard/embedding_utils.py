# Embedding support for text, images, audio
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoFeatureExtractor, ViTModel, ASTForAudioClassification
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchaudio

def generate_text_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    return embeddings



def generate_image_embeddings(image_paths, model_name="google/vit-base-patch16-224"):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(
        model_name, output_hidden_states=True
    )
    embeddings = []
    for i, image_path in enumerate(tqdm(image_paths)):
        try:
            with open(image_path, "rb") as f:
                image = Image.open(f)
                inputs = feature_extractor(images=image.convert("RGB"), return_tensors="pt")
                with torch.nograd():
                    outputs = model(**inputs)
                emb = outputs.last_hidden_state[0,0].cpu().detach().numpy()
                embeddings.append(emb)
                image.close()
        except:
            embeddings.append(None)
            print(f"Could not generate embedding for {image_path}.")
    embeddings = np.array([emb.tolist() if emb is not None else None for emb in embeddings])
    return embeddings



def generate_audio_embeddings(audio_paths, model_name="MIT/ast-finetuned-audioset-10-10-0.4593"):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(
        model_name, output_hidden_states=True
    )
    embeddings = []
    for i, audio_path in enumerate(tqdm(audio_paths)):
        # try: 
        y, sr = torchaudio.load(audio_path)
        y = y.mean(1)
        print(y.shape)
        print(sr)
        inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt")
        with torch.nograd():
            outputs = model(**inputs)
        print(outputs.size())
        emb = outputs.last_hidden_state[0,0].cpu().detach().numpy()
        
        embeddings.append(emb)
        # except:
        #     embeddings.append(None)
        #     print(f"Could not generate embedding for {audio_path}.")
    embeddings = np.array([emb.tolist() if emb is not None else None for emb in embeddings])
    return embeddings