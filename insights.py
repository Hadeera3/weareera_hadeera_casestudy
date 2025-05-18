import re
import emoji
from typing import List, Dict


def post_style_insights(posts: List[str]) -> Dict[str, float]:
    """
    Analyze stylistic features of a list of social media posts.

    Computes average post length (in characters and words), emoji usage rate,
    and average exclamation mark count per post.

    Args:
        posts (List[str]): A list of user post strings.

    Returns:
        Dict[str, float]: A dictionary with the following metrics:
            - "avg_post_length_chars": Average number of characters per post.
            - "avg_post_length_words": Average number of words per post.
            - "emoji_usage_percent": Percentage of posts containing at least one emoji.
            - "avg_exclamations_per_post": Average number of exclamation marks per post.
    """
    if not posts:
        return {}

    total_chars = sum(len(p) for p in posts)
    total_words = sum(len(p.split()) for p in posts)
    emojis = [char for p in posts for char in p if emoji.is_emoji(char)]
    posts_with_emoji = sum(1 for p in posts if any(emoji.is_emoji(c) for c in p))
    exclamations = sum(p.count("!") for p in posts)

    num = len(posts)
    return {
        "avg_post_length_chars": round(total_chars / num, 1),
        "avg_post_length_words": round(total_words / num, 1),
        "emoji_usage_percent": round(100 * posts_with_emoji / num, 1),
        "avg_exclamations_per_post": round(exclamations / num, 2),
    }
