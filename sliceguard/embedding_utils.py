# Embedding support for text, images, audio
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoFeatureExtractor, AutoModel
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torchaudio


def generate_text_embeddings(texts, model_name="all-MiniLM-L6-v2", hf_auth_token=None):
    model = SentenceTransformer(model_name, use_auth_token=hf_auth_token)
    embeddings = model.encode(texts)
    return embeddings


def generate_image_embeddings(
    image_paths, model_name="google/vit-base-patch16-224", hf_auth_token=None
):
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_name, use_auth_token=hf_auth_token
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        model_name, output_hidden_states=True, use_auth_token=hf_auth_token
    ).to(device)
    embeddings = []
    for i, image_path in enumerate(tqdm(image_paths)):
        try:
            with open(image_path, "rb") as f:
                image = Image.open(f)
                inputs = feature_extractor(
                    images=image.convert("RGB"), return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                emb = outputs.last_hidden_state[0, 0].cpu().detach().numpy()
                embeddings.append(emb)
                image.close()
        except:
            embeddings.append(None)
            print(f"Could not generate embedding for {image_path}.")
    embeddings = np.array(
        [emb.tolist() if emb is not None else None for emb in embeddings]
    )
    return embeddings


def generate_audio_embeddings(
    audio_paths,
    model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
    hf_auth_token=None,
):
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_name, use_auth_token=hf_auth_token
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        model_name, output_hidden_states=True, use_auth_token=hf_auth_token
    ).to(device)

    embeddings = []
    for i, audio_path in enumerate(tqdm(audio_paths)):
        # try:
        y, sr = torchaudio.load(audio_path)
        y = y.mean(0)  # convert to mono

        inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state[0, 0].cpu().detach().numpy()

        embeddings.append(emb)
        # except:
        #     embeddings.append(None)
        #     print(f"Could not generate embedding for {audio_path}.")
    embeddings = np.array(
        [emb.tolist() if emb is not None else None for emb in embeddings]
    )
    return embeddings
