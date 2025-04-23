import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List, Dict, Any, Optional

# Load the spaCy model only once
_nlp = None

def get_nlp_model():
    """Lazy-load the spaCy model when needed."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("es_core_news_md")
    return _nlp

def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two texts using TF-IDF.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        float: Cosine similarity score (0-1)
    """
    try:
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w\-/]*\b")
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return cosine_sim[0][0]
    except:
        return 0.0

def calculate_levenshtein_distance(s1: str, s2: str, show_progress: bool = False) -> int:
    """
    Calculate Levenshtein distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        show_progress: Whether to show a progress bar (for long strings)
        
    Returns:
        int: Levenshtein distance
    """
    from tqdm import tqdm
    
    # Swap if s1 is shorter than s2
    if len(s1) < len(s2):
        s1, s2 = s2, s1
        
    # Base case: empty string
    if len(s2) == 0:
        return len(s1)

    # Initialize the first row
    previous_row = list(range(len(s2) + 1))
    
    # Iterate through the first string
    iterable = tqdm(s1) if show_progress else s1
    for i, c1 in enumerate(iterable):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_inverse_levenshtein(distance: int) -> float:
    """
    Calculate inverse Levenshtein score (1/distance).
    
    Args:
        distance: Levenshtein distance
        
    Returns:
        float: Inverse distance (1.0 if distance is 0)
    """
    return 1.0 if distance == 0 else 1.0 / distance

def calculate_embedding_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between texts using spaCy embeddings.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        float: Similarity score (0-1)
    """
    nlp = get_nlp_model()
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)