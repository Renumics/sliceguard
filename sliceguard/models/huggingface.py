from typing import List
from functools import partial
import numpy as np
import pandas as pd
import datasets
from ..embeddings import get_embedding_imports, generate_image_embeddings


def _check_training_imports():
    try:
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
            DefaultDataCollator,
            TrainingArguments,
            Trainer,
        )
        import torch
    except ImportError:
        raise Warning(
            'Optional dependency required! (pip install "sliceguard[embedding]")'
        )

    return


def _transform(example_batch, image_processor):
    inputs = image_processor(
        [x.convert("RGB") for x in example_batch["image"]], return_tensors="pt"
    )
    inputs["label"] = example_batch["label"]
    return inputs


def load_image_preprocessor(model_name, hf_auth_token=None):
    """Load an image preprocessor for the model. This will resize the images to the correct size for the model."""
    from transformers import AutoImageProcessor

    checkpoint = model_name
    image_processor = AutoImageProcessor.from_pretrained(
        checkpoint, use_auth_token=hf_auth_token
    )
    return image_processor


def load_model(model_name, num_labels, model_architecture, hf_auth_token=None):
    """Load the pre-trained model with AutoModelForImageClassification. Specify number of labels."""
    checkpoint = model_name
    model = model_architecture.from_pretrained(
        checkpoint, num_labels=num_labels, use_auth_token=hf_auth_token
    )
    return model


def finetune_image_classifier(
    df,
    model_name="google/vit-base-patch16-224-in21k",
    model_architecture=None,
    output_model_folder=None,
    epochs=5,
    hf_auth_token=None,
):
    # TODO: Full support of top-level arguments.
    _check_training_imports()

    if model_architecture is None:
        from transformers import AutoModelForImageClassification

        model_architecture = AutoModelForImageClassification

    import torch
    from transformers import DefaultDataCollator, TrainingArguments, Trainer

    ds = datasets.Dataset.from_pandas(df)

    image_processor = load_image_preprocessor(model_name, hf_auth_token=hf_auth_token)
    transform_with_processor = partial(_transform, image_processor=image_processor)
    prepared_ds = ds.cast_column("image", datasets.Image())
    prepared_ds = prepared_ds.with_transform(transform_with_processor)
    model = load_model(
        model_name,
        num_labels=df["label"].nunique(),
        model_architecture=model_architecture,
        hf_auth_token=hf_auth_token,
    )

    training_args = TrainingArguments(
        output_dir=output_model_folder,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,  # use 0.04 for testing with a few frames. Use higher values for longer movies
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DefaultDataCollator(),
        train_dataset=prepared_ds,
        eval_dataset=prepared_ds,
        tokenizer=image_processor,
        # callbacks=[PrinterCallback],
    )

    train_results = trainer.train()
    model.save_pretrained(output_model_folder)
    image_processor.save_pretrained(output_model_folder)


def generate_image_pred_probs_embeddings(
    image_paths,
    model_name="google/vit-base-patch16-224",
    hf_num_proc=None,
    hf_batch_size=1,
    hf_auth_token=None,
    return_embeddings=True,
) -> (List, List, List):
    probs = generate_image_probabilites(
        image_paths, model_name, None, hf_num_proc, hf_batch_size
    ).tolist()
    if return_embeddings:
        embeddings = generate_image_embeddings(
            image_paths, model_name, None, hf_num_proc, hf_batch_size
        ).tolist()
    else:
        embeddings = None
    preds = np.argmax(probs, axis=1).tolist()
    return preds, probs, embeddings


def _extract_probabilities_image(model, feature_extractor, col_name="image"):
    """Utility to compute probabilites for images."""

    _, _, _, torch = get_embedding_imports()

    device = model.device

    def pp(batch):
        images = batch[
            col_name
        ]  # not sure if this is smart. probably some feature extractors take multiple modalities.
        for i in range(len(images)):
            if images[i].mode != "RGB":
                images[i] = images[i].convert("RGB")
        inputs = feature_extractor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = (
                torch.nn.functional.softmax(outputs.logits, dim=-1)
                .detach()
                .cpu()
                .numpy()
            )

        return {"probabilities": probabilities}

    return pp


def generate_image_probabilites(
    image_paths,
    model_name="google/vit-base-patch16-224",
    hf_auth_token=None,
    hf_num_proc=None,
    hf_batch_size=1,
):
    _, _, _, torch = get_embedding_imports()
    from transformers import ViTFeatureExtractor, ViTForImageClassification

    feature_extractor = ViTFeatureExtractor.from_pretrained(
        model_name, use_auth_token=hf_auth_token
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Probability computation on {device} with batch size {hf_batch_size} and multiprocessing {hf_num_proc}."
    )

    model = ViTForImageClassification.from_pretrained(
        model_name, output_hidden_states=True, use_auth_token=hf_auth_token
    ).to(device)

    df = pd.DataFrame(data={"image": image_paths})
    dataset = datasets.Dataset.from_pandas(df).cast_column("image", datasets.Image())

    extract_fn = _extract_probabilities_image(model, feature_extractor, "image")

    if hf_num_proc is not None and hf_num_proc > 1:
        set_start_method("spawn", force=True)

    updated_dataset = dataset.map(
        extract_fn,
        batched=True,
        batch_size=hf_batch_size,
        num_proc=hf_num_proc,
        remove_columns="image",
    )  # batches has to be true in general, the batch size could be varied, also multiprocessing could be applied

    df_updated = updated_dataset.to_pandas()

    probabilities = np.array(
        [
            emb.tolist() if emb is not None else None
            for emb in df_updated["probabilities"].values
        ]
    )

    return probabilities
