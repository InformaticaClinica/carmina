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

def calculate_metrics(line_a:str, line_b:str, metrics: Dict[str, Any]) -> Dict[str, float]:
    total = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }
    for line_id, line_data in metrics.items():
        # Skip line_0 as it's debug only and shouldn't participate in metrics
        if line_id == "line_0":
            continue
        total["tp"] += line_data["metrics"]["tp"]
        total["fp"] += line_data["metrics"]["fp"]
        total["fn"] += line_data["metrics"]["fn"]
    total["precision"] = calculate_precision(total["tp"], total["fp"])
    total["recall"] = calculate_recall(total["tp"], total["fn"])
    total["f1"] = calculate_f1(total["precision"], total["recall"])
    total["cosine_similarity"] = calculate_cosine_similarity(line_a, line_b)
    total["levenshtein_distance"] = calculate_levenshtein_distance(line_a, line_b)
    total["inv_levenshtein"] = calculate_inverse_levenshtein(total["levenshtein_distance"])
    return total


def calculate_average_metrics(metrics: Dict[str, Any], typefile: str) -> Dict[str, float]:
    """
    Calculate average metrics from a dictionary of metrics.
    
    Args:
        metrics: Dictionary of metrics
    
    Returns:
        Dict[str, float]: Dictionary of average metrics
    """
    total = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "cosine_sim": 0.0,
        "levenshtein_distance": 0.0,
        "inv_levenshtein": 0.0,
        "overall": 0.0,
    }

    count = len(metrics)
    for file in metrics:
        file_data = file.get(typefile) if isinstance(file, dict) else None
        if not file_data or not isinstance(file_data, dict):
            count = max(count - 1, 0)
            continue
        file_metrics = file_data.get("metrics")
        if not file_metrics or not isinstance(file_metrics, dict):
            count = max(count - 1, 0)
            continue
        total["precision"] += file_metrics.get("precision", 0.0)
        total["recall"] += file_metrics.get("recall", 0.0)
        total["f1"] += file_metrics.get("f1", 0.0)
        total["cosine_sim"] += file_metrics.get("cosine_similarity", 0.0)
        total["levenshtein_distance"] += file_metrics.get("levenshtein_distance", 0.0)
        total["inv_levenshtein"] += file_metrics.get("inv_levenshtein", 0.0)
    
    total["precision"] /= count if count > 0 else 1
    total["recall"] /= count if count > 0 else 1
    total["f1"] /= count if count > 0 else 1
    total["cosine_sim"] /= count if count > 0 else 1
    total["levenshtein_distance"] /= count if count > 0 else 1
    total["inv_levenshtein"] /= count if count > 0 else 1
    total["overall"] = total["precision"] + total["recall"] + total["f1"] + \
                        total["cosine_sim"] + total["inv_levenshtein"]
    return total


def evaluate_array(gt_record: List[str], 
                           pred_record: List[str]):
    """
    Evaluate PHI identification performance.
    """
    tp, fp, fn = calculate_positives_and_negatives(gt_record, pred_record)
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    f1 = calculate_f1(precision, recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def get_all_metrics(tags_a: List[str], tags_b: List[str]) -> Dict[str, Any]:
    """
    Get all metrics for a pair of lines.
    
    Args:
        line_a: First line
        line_b: Second line
        tags_a: Tags from the first line
        tags_b: Tags from the second line
        
    Returns:
        Dict[str, Any]: Dictionary of metrics
    """
    metrics = evaluate_array(tags_a, tags_b)
    return metrics


def evaluate_identification(ground_truth_records: List[str], 
                           prediction_records: List[str]) -> Dict[str, float]:
    """
    Evaluate PHI identification performance.
    
    Args:
        ground_truth_records: List of ground truth identified entities
        prediction_records: List of predicted identified entities
        
    Returns:
        Dict[str, float]: Dictionary of identification metrics
    """
    # Use the existing evaluate_array function
    result = evaluate_array(ground_truth_records, prediction_records)
    
    # Return metrics with the expected prefixes for identification
    return {
        "identification_precision": result["precision"],
        "identification_recall": result["recall"],
        "identification_f1": result["f1"],
        "identification_tp": result["tp"],
        "identification_fp": result["fp"],
        "identification_fn": result["fn"]
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
        Dict[str, float]: Dictionary of label metrics
    """
    total_cosine = 0.0
    total_levenshtein = 0.0
    total_inv_levenshtein = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    count = len(ground_truth_texts)
    
    for i in range(count):
        # Text similarity metrics
        if i < len(prediction_texts):
            total_cosine += calculate_cosine_similarity(ground_truth_texts[i], prediction_texts[i])
            total_levenshtein += calculate_levenshtein_distance(ground_truth_texts[i], prediction_texts[i])
            total_inv_levenshtein += calculate_inverse_levenshtein(
                calculate_levenshtein_distance(ground_truth_texts[i], prediction_texts[i])
            )
        
        # Label classification metrics
        if i < len(ground_truth_labels) and i < len(prediction_labels):
            gt_labels = ground_truth_labels[i] if isinstance(ground_truth_labels[i], list) else [ground_truth_labels[i]]
            pred_labels = prediction_labels[i] if isinstance(prediction_labels[i], list) else [prediction_labels[i]]
            
            result = evaluate_array(gt_labels, pred_labels)
            total_precision += result["precision"]
            total_recall += result["recall"]
            total_f1 += result["f1"]
            total_tp += result["tp"]
            total_fp += result["fp"]
            total_fn += result["fn"]
    
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
        avg_tp = avg_fp = avg_fn = 0
    
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


# def evaluate_identification(ground_truth_records: List[List[str]], 
#                            prediction_records: List[List[str]],
#                            filenames: List[str],
#                            language: List[str]) -> Dict[str, Any]:
#     """
#     Evaluate PHI identification performance.
    
#     Args:
#         ground_truth_records: List of ground truth records (list of lists)
#         prediction_records: List of predicted records (list of lists)
        
#     Returns:
#         Dict[str, Any]: Dictionary of metrics with averages and per-file details
#     """
#     total_precision = 0.0
#     total_recall = 0.0
#     total_f1 = 0.0
#     total_tp = 0
#     total_fp = 0
#     total_fn = 0
#     count = 0
#     per_file_metrics = []
    
#     for gt_record, pred_record in zip(ground_truth_records, prediction_records):
#         # Calculate metrics for each file/record
#         tp, fp, fn = calculate_positives_and_negatives(gt_record, pred_record)
#         precision = calculate_precision(tp, fp)
#         recall = calculate_recall(tp, fn)
#         f1 = calculate_f1(precision, recall)
        
#         # Add to totals
#         total_precision += precision
#         total_recall += recall
#         total_f1 += f1
#         total_tp += tp
#         total_fp += fp
#         total_fn += fn
#         count += 1
        
#         # Store per-file metrics
#         per_file_metrics.append({
#             "filename": filenames[count - 1] if filenames else None,
#             "language": language[count - 1] if language else None,
#             "tp": tp,
#             "fp": fp,
#             "fn": fn,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#             "labels": [gt_record, pred_record]
#         })
    
#     # Calculate averages
#     avg_precision = total_precision / count if count > 0 else 0
#     avg_recall = total_recall / count if count > 0 else 0
#     avg_f1 = total_f1 / count if count > 0 else 0
#     avg_tp = total_tp / count if count > 0 else 0
#     avg_fp = total_fp / count if count > 0 else 0
#     avg_fn = total_fn / count if count > 0 else 0
    
#     return {
#         "identification_precision": avg_precision,
#         "identification_recall": avg_recall,
#         "identification_f1": avg_f1,
#         "identification_tp": avg_tp,
#         "identification_fp": avg_fp,
#         "identification_fn": avg_fn,
#         "identification_per_file": per_file_metrics
#     }

# def evaluate_label(ground_truth_texts: List[str], 
#                   prediction_texts: List[str],
#                   ground_truth_labels: List[List[str]],
#                   prediction_labels: List[List[str]],
#                   filenames: List[str],
#                   language: List[str]) -> Dict[str, float]:
#     """
#     Evaluate text label quality.
    
#     Args:
#         ground_truth_texts: List of ground truth texts
#         prediction_texts: List of predicted texts
#         ground_truth_labels: List of ground truth labels (list of lists)
#         prediction_labels: List of predicted labels (list of lists)
        
#     Returns:
#         Dict[str, float]: Dictionary of metrics
#     """
#     total_cosine = 0.0
#     total_levenshtein = 0
#     total_inv_levenshtein = 0.0
#     total_precision = 0.0
#     total_recall = 0.0
#     total_tp = 0.0
#     total_fp = 0.0
#     total_fn = 0.0
#     total_f1 = 0.0
#     count = 0
#     per_file_metrics = []
    
#     for gt_text, pred_text, gt_labels, pred_labels in zip(
#             ground_truth_texts, prediction_texts, 
#             ground_truth_labels, prediction_labels):
#         # Calculate similarity metrics for texts
#         cosine_sim = calculate_cosine_similarity(gt_text, pred_text)
#         levenshtein = calculate_levenshtein_distance(gt_text, pred_text)
#         inv_levenshtein = calculate_inverse_levenshtein(levenshtein)
        
#         # Calculate classification metrics for each document's labels
#         tp, fp, fn = calculate_positives_and_negatives(gt_labels, pred_labels)
#         precision = calculate_precision(tp, fp)
#         recall = calculate_recall(tp, fn)
#         f1 = calculate_f1(precision, recall)
        
#         total_cosine += cosine_sim
#         total_levenshtein += levenshtein
#         total_inv_levenshtein += inv_levenshtein
#         total_precision += precision
#         total_recall += recall
#         total_f1 += f1
#         count += 1
#         total_tp += tp
#         total_fp += fp
#         total_fn += fn
#         per_file_metrics.append({
#             "filename": filenames[count - 1] if filenames else None,
#             "language": language[count - 1] if language else None,
#             "cosine_sim": cosine_sim,
#             "levenshtein_distance": levenshtein,
#             "inv_levenshtein": inv_levenshtein,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#             "tp": tp,
#             "fp": fp,
#             "fn": fn,
#             "labels": [gt_labels, pred_labels],
#             "texts": [gt_text, pred_text]
#         })

        
#     # Calculate averages
#     if count > 0:
#         avg_cosine = total_cosine / count
#         avg_levenshtein = total_levenshtein / count
#         avg_inv_levenshtein = total_inv_levenshtein / count
#         avg_precision = total_precision / count
#         avg_recall = total_recall / count
#         avg_f1 = total_f1 / count
#         avg_tp = total_tp / count
#         avg_fp = total_fp / count
#         avg_fn = total_fn / count
#     else:
#         avg_cosine = avg_levenshtein = avg_inv_levenshtein = 0
#         avg_precision = avg_recall = avg_f1 = 0
    
#     return {
#         "label_cosine_sim": avg_cosine,
#         "label_levenshtein": avg_levenshtein,
#         "label_inv_levenshtein": avg_inv_levenshtein,
#         "label_precision": avg_precision,
#         "label_recall": avg_recall,
#         "label_f1": avg_f1,
#         "label_overall": avg_cosine + avg_inv_levenshtein + avg_f1 + avg_precision + avg_recall,
#         "label_tp": avg_tp,
#         "label_fp": avg_fp,
#         "label_fn": avg_fn,
#         "label_per_file": per_file_metrics
#     }

# def evaluate_text_pair(ground_truth: str, 
#                      generated: str, 
#                      classification: bool = True,
#                      cosine_sim: bool = True, 
#                      levenshtein: bool = True,
#                      start_time: Optional[float] = None) -> Dict[str, Any]:
#     """
#     Evaluate a pair of texts and calculate metrics.
    
#     Args:
#         ground_truth: Ground truth text
#         generated: Generated text
#         classification: Whether to calculate classification metrics
#         cosine_sim: Whether to calculate cosine similarity
#         levenshtein: Whether to calculate Levenshtein distance
#         start_time: Start time for timing calculation
        
#     Returns:
#         Dict[str, Any]: Dictionary of metrics
#     """
#     results = {}
    
#     # Extract labels if classification needed
#     if classification:
#         ground_truth_labels = extract_labels_from_masked_text(ground_truth)
#         generated_labels = extract_labels_from_masked_text(generated)
#         tp, fp, fn = calculate_positives_and_negatives(ground_truth_labels, generated_labels)
        
#         precision = calculate_precision(tp, fp)
#         recall = calculate_recall(tp, fn)
#         f1 = calculate_f1(precision, recall)
        
#         results.update({
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#             "labels": [ground_truth_labels, generated_labels]
#         })
    
#     # Calculate text similarity
#     if cosine_sim:
#         results["cosine_sim"] = calculate_cosine_similarity(ground_truth, generated)
    
#     # Calculate Levenshtein metrics
#     if levenshtein:
#         results["levenshtein_distance"] = calculate_levenshtein_distance(ground_truth, generated)
#         results["inv_levenshtein"] = calculate_inverse_levenshtein(results["levenshtein_distance"])
    
#     # Calculate overall score if all metrics are available
#     if classification and cosine_sim and levenshtein:
#         results["overall"] = results["precision"] + results["recall"] + results["f1"] + \
#                             results["cosine_sim"] + results["inv_levenshtein"]
    
#     # Add timing information if provided
#     if start_time is not None:
#         results["time"] = time.time() - start_time
    
#     return results

