from typing import List, Tuple
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline, Pipeline

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
ZERO_SHOT_MODEL_NAME = "facebook/bart-large-mnli"
TOP_K = 5
ALPHA_DEFAULT = 0.3
BETA_DEFAULT = 0.7


def load_data(path: str) -> List[dict]:
    """
    Load a JSON file into a list of dictionaries.

    Args:
        path (str): Path to the JSON file.

    Returns:
        List[dict]: A list of records (each as a dict) parsed from the file.
    """

    return pd.read_json(path).to_dict(orient="records")


def init_models() -> Tuple[SentenceTransformer, Pipeline]:
    """
    Initialize and return the embedding and zero-shot classification models.
    Embedding model is used for semantic similarity (via SentenceTransformer),
    Zero-shot model performs classification with natural language prompts.

    Returns:
        Tuple[
            SentenceTransformer,
            Pipeline
        ]
    """
    return (
        SentenceTransformer(EMBEDDING_MODEL_NAME),
        pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL_NAME, device=1),
    )
