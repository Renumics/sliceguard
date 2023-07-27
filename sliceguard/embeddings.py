# Embedding support for text, images, audio
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoFeatureExtractor, AutoModel
import datasets
import numpy as np
import torch


def generate_text_embeddings(
    texts,
    model_name="all-MiniLM-L6-v2",
    hf_auth_token=None,
    hf_num_proc=None,
    hf_batch_size=1,
):
    if hf_num_proc:
        print(
            "Warning: Multiprocessing cannot be used in generating text embeddings yet."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Embedding computation on {device} with batch size {hf_batch_size} and multiprocessing {hf_num_proc}."
    )

    model = SentenceTransformer(model_name, device=device, use_auth_token=hf_auth_token)
    embeddings = model.encode(texts, batch_size=hf_batch_size)
    return embeddings


def _extract_embeddings_images(model, feature_extractor, col_name="image"):
    """Utility to compute embeddings for images."""
    device = model.device

    def pp(batch):
        images = batch[
            col_name
        ]  # not sure if this is smart. probably some feature extractors take multiple modalities.
        inputs = feature_extractor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state[:, 0].detach().cpu()

        return {"embedding": embeddings}

    return pp


def generate_image_embeddings(
    image_paths,
    model_name="google/vit-base-patch16-224",
    hf_auth_token=None,
    hf_num_proc=None,
    hf_batch_size=1,
):
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_name, use_auth_token=hf_auth_token
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Embedding computation on {device} with batch size {hf_batch_size} and multiprocessing {hf_num_proc}."
    )

    model = AutoModel.from_pretrained(
        model_name, output_hidden_states=True, use_auth_token=hf_auth_token
    ).to(device)

    df = pd.DataFrame(data={"image": image_paths})
    dataset = datasets.Dataset.from_pandas(df).cast_column("image", datasets.Image())

    extract_fn = _extract_embeddings_images(
        model.to(device), feature_extractor, "image"
    )
    updated_dataset = dataset.map(
        extract_fn, batched=True, batch_size=hf_batch_size, num_proc=hf_num_proc
    )  # batches has to be true in general, the batch size could be varied, also multiprocessing could be applied

    df_updated = updated_dataset.to_pandas()

    embeddings = np.array(
        [
            emb.tolist() if emb is not None else None
            for emb in df_updated["embedding"].values
        ]
    )

    return embeddings


def _extract_embeddings_audios(model, feature_extractor, col_name="audio"):
    """Utility to compute embeddings for audios."""
    device = model.device

    def pp(batch):
        audios = batch[
            col_name
        ]  # not sure if this is smart. probably some feature extractors take multiple modalities.
        sampling_rates = np.array([a["sampling_rate"] for a in audios])
        if not np.all(sampling_rates[0] == sampling_rates):
            raise RuntimeError(
                "If batching is active, sampling rates should be the same for all batch samples."
            )
        inputs = feature_extractor(
            raw_speech=[a["array"] for a in audios],
            sampling_rate=sampling_rates[0],
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state[:, 0].detach().cpu()

        return {"embedding": embeddings}

    return pp


def generate_audio_embeddings(
    audio_paths,
    model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
    hf_auth_token=None,
    hf_num_proc=None,
    hf_batch_size=1,
):
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_name, use_auth_token=hf_auth_token
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Embedding computation on {device} with batch size {hf_batch_size} and multiprocessing {hf_num_proc}."
    )

    model = AutoModel.from_pretrained(
        model_name, output_hidden_states=True, use_auth_token=hf_auth_token
    ).to(device)

    df = pd.DataFrame(data={"audio": audio_paths})

    dataset = datasets.Dataset.from_pandas(df).cast_column("audio", datasets.Audio())

    extract_fn = _extract_embeddings_audios(
        model.to(device), feature_extractor, "audio"
    )
    updated_dataset = dataset.map(
        extract_fn, batched=True, batch_size=hf_batch_size, num_proc=hf_num_proc
    )  # batches has to be true in general, the batch size could be varied

    df_updated = updated_dataset.to_pandas()

    embeddings = np.array(
        [
            emb.tolist() if emb is not None else None
            for emb in df_updated["embedding"].values
        ]
    )

    return embeddings
