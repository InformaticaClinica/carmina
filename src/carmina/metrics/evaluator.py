import re
import time
from typing import List, Dict, Any, Tuple, Optional, Union
from .classification import (
    calculate_precision, calculate_recall, calculate_f1,
    calculate_positives_and_negatives
)
from .similarity import (
    calculate_cosine_similarity, calculate_levenshtein_distance,
    calculate_inverse_levenshtein
)
from .extractors import extract_labels_from_masked_text

def evaluate_identification(ground_truth_records: List[str], 
                           prediction_records: List[str]) -> Dict[str, float]:
    """
    Evaluate PHI identification performance.
    
    Args:
        ground_truth: Array with ground truth labels
        predictions: Array with predicted labels
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    # Calculate metrics directly from the arrays
    tp, fp, fn = calculate_positives_and_negatives(ground_truth_records, prediction_records)
    
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    f1 = calculate_f1(precision, recall)
    
    return {
        "identification_precision": precision,
        "identification_recall": recall,
        "identification_f1": f1,
        "identification_tp": tp,
        "identification_fp": fp,
        "identification_fn": fn
    }

def evaluate_substitution(ground_truth_records: List[Dict[str, Any]], 
                         prediction_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate text substitution quality.
    
    Args:
        ground_truth_records: List of ground truth records
        prediction_records: List of predicted records
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    total_cosine = 0.0
    total_levenshtein = 0
    total_inv_levenshtein = 0.0
    count = 0
    
    for gt_record, pred_record in zip(ground_truth_records, prediction_records):
        if 'text' in gt_record and 'text' in pred_record:
            gt_text = gt_record['text']
            pred_text = pred_record['text']
            
            # Calculate similarity metrics
            cosine_sim = calculate_cosine_similarity(gt_text, pred_text)
            levenshtein = calculate_levenshtein_distance(gt_text, pred_text)
            inv_levenshtein = calculate_inverse_levenshtein(levenshtein)
            
            total_cosine += cosine_sim
            total_levenshtein += levenshtein
            total_inv_levenshtein += inv_levenshtein
            count += 1
    
    # Calculate averages
    avg_cosine = total_cosine / count if count > 0 else 0
    avg_levenshtein = total_levenshtein / count if count > 0 else 0
    avg_inv_levenshtein = total_inv_levenshtein / count if count > 0 else 0
    
    return {
        "substitution_cosine_sim": avg_cosine,
        "substitution_levenshtein": avg_levenshtein,
        "substitution_inv_levenshtein": avg_inv_levenshtein,
        "substitution_overall": avg_cosine + avg_inv_levenshtein
    }

def evaluate_text_pair(ground_truth: str, 
                     generated: str, 
                     classification: bool = True,
                     cosine_sim: bool = True, 
                     levenshtein: bool = True,
                     start_time: Optional[float] = None) -> Dict[str, Any]:
    """
    Evaluate a pair of texts and calculate metrics.
    
    Args:
        ground_truth: Ground truth text
        generated: Generated text
        classification: Whether to calculate classification metrics
        cosine_sim: Whether to calculate cosine similarity
        levenshtein: Whether to calculate Levenshtein distance
        start_time: Start time for timing calculation
        
    Returns:
        Dict[str, Any]: Dictionary of metrics
    """
    results = {}
    
    # Extract labels if classification needed
    if classification:
        ground_truth_labels = extract_labels_from_masked_text(ground_truth)
        generated_labels = extract_labels_from_masked_text(generated)
        tp, fp, fn = calculate_positives_and_negatives(ground_truth_labels, generated_labels)
        
        precision = calculate_precision(tp, fp)
        recall = calculate_recall(tp, fn)
        f1 = calculate_f1(precision, recall)
        
        results.update({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "labels": [ground_truth_labels, generated_labels]
        })
    
    # Calculate text similarity
    if cosine_sim:
        results["cosine_sim"] = calculate_cosine_similarity(ground_truth, generated)
    
    # Calculate Levenshtein metrics
    if levenshtein:
        results["levenshtein_distance"] = calculate_levenshtein_distance(ground_truth, generated)
        results["inv_levenshtein"] = calculate_inverse_levenshtein(results["levenshtein_distance"])
    
    # Calculate overall score if all metrics are available
    if classification and cosine_sim and levenshtein:
        results["overall"] = results["precision"] + results["recall"] + results["f1"] + \
                            results["cosine_sim"] + results["inv_levenshtein"]
    
    # Add timing information if provided
    if start_time is not None:
        results["time"] = time.time() - start_time
    
    return results