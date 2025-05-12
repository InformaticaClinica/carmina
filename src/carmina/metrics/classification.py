from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

def calculate_precision(true_positives: int, false_positives: int) -> float:
    """
    Calculate precision metric.
    
    Args:
        true_positives: Number of true positives
        false_positives: Number of false positives
        
    Returns:
        float: Precision score (0-1)
    """
    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

def calculate_recall(true_positives: int, false_negatives: int) -> float:
    """
    Calculate recall metric.
    
    Args:
        true_positives: Number of true positives
        false_negatives: Number of false negatives
        
    Returns:
        float: Recall score (0-1)
    """
    return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

def calculate_f1(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.
    
    Args:
        precision: Precision score
        recall: Recall score
        
    Returns:
        float: F1 score (0-1)
    """
    if precision + recall <= 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def create_distance_matrix(str1: str, str2: str) -> pd.DataFrame:
    """
    Create a matrix of edit distances between words in two strings.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        pd.DataFrame: Distance matrix
    """
    # Split the strings into words
    str1_words = str1.split(' ')
    str2_words = str2.split(' ')

    # Initialize the matrix
    m, n = len(str1_words), len(str2_words)
    matrix = np.zeros((m + 1, n + 1), dtype=int)

    # Fill the first row and first column
    for i in range(1, m + 1):
        matrix[i][0] = i
    for j in range(1, n + 1):
        matrix[0][j] = j

    # Fill the matrix with edit distances
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1_words[i - 1] == str2_words[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                matrix[i][j] = 1 + min(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1])

    # Create a DataFrame to include the words in the headers
    df = pd.DataFrame(matrix, index=[""] + str1_words, columns=[""] + str2_words)
    return df

def trace_edit_path(matrix: pd.DataFrame, str1_words: List[str], str2_words: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Trace the optimal edit path through the distance matrix.
    
    Args:
        matrix: Distance matrix
        str1_words: Words from first string
        str2_words: Words from second string
        
    Returns:
        Tuple[List[str], List[str], List[str]]: Deleted words, added words, unchanged words
    """
    i, j = len(str1_words), len(str2_words)

    deleted = []
    added = []
    unchanged = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and str1_words[i - 1] == str2_words[j - 1]:
            unchanged.append(str1_words[i - 1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or matrix.iloc[i, j] == matrix.iloc[i - 1, j] + 1):
            deleted.append(str1_words[i - 1])
            i -= 1
        elif j > 0 and (i == 0 or matrix.iloc[i, j] == matrix.iloc[i, j - 1] + 1):
            added.append(str2_words[j - 1])
            j -= 1
        else:
            deleted.append(str1_words[i - 1])
            added.append(str2_words[j - 1])
            i -= 1
            j -= 1

    return deleted[::-1], added[::-1], unchanged[::-1]  # Reverse to preserve order

def calculate_positives_and_negatives(ground_truth_labels: List[str], generated_labels: List[str]) -> Tuple[int, int, int]:
    """
    Calculate true positives, false positives, and false negatives by comparing label lists.
    
    Args:
        ground_truth_labels: List of ground truth labels
        generated_labels: List of generated labels
        
    Returns:
        Tuple[int, int, int]: True positives, false positives, false negatives
    """
    # Check if inputs are lists
    if not isinstance(ground_truth_labels, list) or not isinstance(generated_labels, list):
        raise TypeError("Both inputs must be lists")

    # Use set operations to find matching and differing labels
    ground_truth_set = set(ground_truth_labels)
    generated_set = set(generated_labels)
    
    true_positives = len(ground_truth_set & generated_set)
    false_positives = len(generated_set - ground_truth_set)
    false_negatives = len(ground_truth_set - generated_set)
    
    return true_positives, false_positives, false_negatives
