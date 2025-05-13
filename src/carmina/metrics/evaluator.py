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

def evaluate_identification(ground_truth_records: List[List[str]], 
                           prediction_records: List[List[str]]) -> Dict[str, Any]:
    """
    Evaluate PHI identification performance.
    
    Args:
        ground_truth_records: List of ground truth records (list of lists)
        prediction_records: List of predicted records (list of lists)
        
    Returns:
        Dict[str, Any]: Dictionary of metrics with averages and per-file details
    """
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    count = 0
    per_file_metrics = []
    
    for gt_record, pred_record in zip(ground_truth_records, prediction_records):
        # Calculate metrics for each file/record
        tp, fp, fn = calculate_positives_and_negatives(gt_record, pred_record)
        precision = calculate_precision(tp, fp)
        recall = calculate_recall(tp, fn)
        f1 = calculate_f1(precision, recall)
        
        # Add to totals
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_tp += tp
        total_fp += fp
        total_fn += fn
        count += 1
        
        # Store per-file metrics
        per_file_metrics.append({
            "tp": tp,
            "fp": fp,
            "fn": fn
        })
    
    # Calculate averages
    avg_precision = total_precision / count if count > 0 else 0
    avg_recall = total_recall / count if count > 0 else 0
    avg_f1 = total_f1 / count if count > 0 else 0
    avg_tp = total_tp / count if count > 0 else 0
    avg_fp = total_fp / count if count > 0 else 0
    avg_fn = total_fn / count if count > 0 else 0
    
    return {
        "identification_precision": avg_precision,
        "identification_recall": avg_recall,
        "identification_f1": avg_f1,
        "identification_tp": avg_tp,
        "identification_fp": avg_fp,
        "identification_fn": avg_fn,
        "identification_per_file": per_file_metrics
    }

def evaluate_label(ground_truth_texts: List[str], 
                  prediction_texts: List[str],
                  ground_truth_labels: List[List[str]],
                  prediction_labels: List[List[str]]) -> Dict[str, float]:
    """
    Evaluate text label quality.
    
    Args:
        ground_truth_texts: List of ground truth texts
        prediction_texts: List of predicted texts
        ground_truth_labels: List of ground truth labels (list of lists)
        prediction_labels: List of predicted labels (list of lists)
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    total_cosine = 0.0
    total_levenshtein = 0
    total_inv_levenshtein = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_f1 = 0.0
    count = 0
    
    for gt_text, pred_text, gt_labels, pred_labels in zip(
            ground_truth_texts, prediction_texts, 
            ground_truth_labels, prediction_labels):
        # Calculate similarity metrics for texts
        cosine_sim = calculate_cosine_similarity(gt_text, pred_text)
        levenshtein = calculate_levenshtein_distance(gt_text, pred_text)
        inv_levenshtein = calculate_inverse_levenshtein(levenshtein)
        
        # Calculate classification metrics for each document's labels
        tp, fp, fn = calculate_positives_and_negatives(gt_labels, pred_labels)
        precision = calculate_precision(tp, fp)
        recall = calculate_recall(tp, fn)
        f1 = calculate_f1(precision, recall)
        
        total_cosine += cosine_sim
        total_levenshtein += levenshtein
        total_inv_levenshtein += inv_levenshtein
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        count += 1
        total_tp += tp
        total_fp += fp
        total_fn += fn

        
    # Calculate averages
    if count > 0:
        avg_cosine = total_cosine / count
        avg_levenshtein = total_levenshtein / count
        avg_inv_levenshtein = total_inv_levenshtein / count
        avg_precision = total_precision / count
        avg_recall = total_recall / count
        avg_f1 = total_f1 / count
        avg_tp = total_tp / count
        avg_fp = total_fp / count
        avg_fn = total_fn / count
    else:
        avg_cosine = avg_levenshtein = avg_inv_levenshtein = 0
        avg_precision = avg_recall = avg_f1 = 0
    
    return {
        "label_cosine_sim": avg_cosine,
        "label_levenshtein": avg_levenshtein,
        "label_inv_levenshtein": avg_inv_levenshtein,
        "label_precision": avg_precision,
        "label_recall": avg_recall,
        "label_f1": avg_f1,
        "label_overall": avg_cosine + avg_inv_levenshtein + avg_f1 + avg_precision + avg_recall,
        "label_tp": avg_tp,
        "label_fp": avg_fp,
        "label_fn": avg_fn
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