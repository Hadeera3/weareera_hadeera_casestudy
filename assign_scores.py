import numpy as np
from typing import List, Dict, Union
from sentence_transformers import SentenceTransformer
from transformers import Pipeline
from sklearn.metrics.pairwise import cosine_similarity


def encode_text(texts: Union[str, List[str]], model: SentenceTransformer) -> np.ndarray:
    """
    Encode texts into dense vectors using a sentence embedding model.

    Args:
        texts (Union[str, List[str]]): A single string or list of strings to encode.
        model (SentenceTransformer): Preloaded sentence transformer model.

    Returns:
        np.ndarray: A 2D array of embeddings, shape (N, D), where N = number of texts.
    """
    if isinstance(texts, str):
        texts = [texts]
    return model.encode(texts, normalize_embeddings=True)


def embed_user_input(
    bio: str, posts: List[str], model: SentenceTransformer, weight_bio: float = 2.0
) -> np.ndarray:
    """
    Create a single user embedding from bio and posts, with weights for bio.

    Args:
        bio (str): User's profile bio.
        posts (List[str]): List of recent user posts.
        model (SentenceTransformer): Embedding model.
        weight_bio (float): Scaling factor applied to the bio vector (default: 1.0).

    Returns:
        np.ndarray: Averaged user embedding vector of shape (D,).
    """
    texts = [bio.strip()] + posts
    weights = [weight_bio] + [1.0] * len(posts)
    vectors = encode_text(texts, model)
    weighted = np.array([w * v for w, v in zip(weights, vectors)])
    return weighted.mean(axis=0)


def embed_personalities(personas: List[dict], model: SentenceTransformer) -> np.ndarray:
    """
    Embed the descriptions of each personality type into vectors.

    Args:
        personas (List[dict]): List of persona definitions, each with a 'description' field.
        model (SentenceTransformer): Sentence transformer model to use.

    Returns:
        np.ndarray: Matrix of personality embeddings, shape (num_personas, D).
    """
    return encode_text([p["description"] for p in personas], model)


def compute_scores(
    bio: str,
    posts: List[str],
    personalities: List[dict],
    personalities_vectors: np.ndarray,
    embedding_model: SentenceTransformer,
    zero_shot: Pipeline,
    alpha: float,
    beta: float,
    top_k: int,
    weight_bio: float = 2.0,
) -> List[Dict]:
    """
    Compute similarity scores between user input and known personalities using
    both embedding similarity and zero-shot classification, then combine them.

    Args:
        bio (str): User bio text.
        posts (List[str]): User post texts.
        personalities (List[dict]): Persona metadata with 'label' and 'description'.
        personalities_vectors (np.ndarray): Precomputed persona embeddings.
        embedding_model (SentenceTransformer): Embedding model for user input.
        zero_shot (Pipeline): HuggingFace zero-shot classification pipeline.
        alpha (float): Weight for embedding similarity score.
        beta (float): Weight for zero-shot classification score.
        top_k (int): Number of top personas to return.
        weight_bio (float): Optional weight for the bio when embedding.

    Returns:
        List[Dict]: List of top-k personas with `label`, `embed_score`, `zero_shot_score`, and `combined_score`.
    """
    user_vector = embed_user_input(bio, posts, embedding_model, weight_bio)
    sims = cosine_similarity([user_vector], personalities_vectors)[0]
    top_idxs = np.argsort(sims)[::-1][:top_k]
    labels = [personalities[i]["label"] for i in top_idxs]

    zs = zero_shot(bio + " " + " ".join(posts), candidate_labels=labels)
    zero_shot_scores = dict(zip(zs["labels"], zs["scores"]))

    return sorted(
        [
            {
                "label": label,
                "embed_score": round(float(sims[i]), 3),
                "zero_shot_score": round(float(zero_shot_scores.get(label, 0.0)), 3),
                "combined_score": round(
                    alpha * sims[i] + beta * zero_shot_scores.get(label, 0.0), 3
                ),
            }
            for i, label in zip(top_idxs, labels)
        ],
        key=lambda x: x["zero_shot_score"],
        reverse=True,
    )


def recommend_products(
    posts: List[str],
    top_personas: List[str],
    product_catalog: Dict[str, List[Dict]],
    model: SentenceTransformer,
    top_k: int,
) -> List[Dict]:
    """
    Recommend relevant products based on user's post content and top persona matches.

    Args:
        posts (List[str]): Recent user posts.
        top_personas (List[str]): Labels of top persona predictions.
        product_catalog (Dict[str, List[Dict]]): Mapping from persona label to product list.
        model (SentenceTransformer): Embedding model for post and product encoding.
        top_k (int): Number of top product recommendations to return.

    Returns:
        List[Dict]: A ranked list of product matches with fields:
            - product (str)
            - category (str)
            - score (float)
    """
    user_vector = encode_text(" ".join(posts), model)[0]
    products = [p for persona in top_personas for p in product_catalog.get(persona, [])]
    if not products:
        return []

    product_texts = [f"{p['product']} - {p['category']}" for p in products]
    product_vecs = encode_text(product_texts, model)
    scores = cosine_similarity([user_vector], product_vecs)[0]

    return sorted(
        [
            {"product": p["product"], "category": p["category"], "score": round(s, 3)}
            for p, s in zip(products, scores)
        ],
        key=lambda x: x["score"],
        reverse=True,
    )[:top_k]
