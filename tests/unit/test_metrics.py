"""
Unit tests for metrics functionality.
"""
import pytest
from src.carmina.metrics.classification import (
    calculate_precision, calculate_recall, calculate_f1,
    calculate_positives_and_negatives
)
from src.carmina.metrics.similarity import (
    calculate_cosine_similarity, calculate_levenshtein_distance,
    calculate_inverse_levenshtein
)
from src.carmina.metrics.evaluator import evaluate_identification, evaluate_label
from src.carmina.metrics.extractors import extract_labels_from_masked_text


@pytest.mark.unit
class TestClassificationMetrics:
    """Test classification metrics functionality."""
    
    def test_calculate_precision_perfect(self):
        """Test precision calculation with perfect score."""
        precision = calculate_precision(true_positives=10, false_positives=0)
        assert precision == 1.0
    
    def test_calculate_precision_zero_tp(self):
        """Test precision calculation with zero true positives."""
        precision = calculate_precision(true_positives=0, false_positives=5)
        assert precision == 0.0
    
    def test_calculate_precision_mixed(self):
        """Test precision calculation with mixed results."""
        precision = calculate_precision(true_positives=7, false_positives=3)
        assert precision == 0.7
    
    def test_calculate_recall_perfect(self):
        """Test recall calculation with perfect score."""
        recall = calculate_recall(true_positives=10, false_negatives=0)
        assert recall == 1.0
    
    def test_calculate_recall_zero_tp(self):
        """Test recall calculation with zero true positives."""
        recall = calculate_recall(true_positives=0, false_negatives=5)
        assert recall == 0.0
    
    def test_calculate_recall_mixed(self):
        """Test recall calculation with mixed results."""
        recall = calculate_recall(true_positives=8, false_negatives=2)
        assert recall == 0.8
    
    def test_calculate_f1_perfect(self):
        """Test F1 calculation with perfect scores."""
        f1 = calculate_f1(precision=1.0, recall=1.0)
        assert f1 == 1.0
    
    def test_calculate_f1_zero(self):
        """Test F1 calculation with zero scores."""
        f1 = calculate_f1(precision=0.0, recall=0.0)
        assert f1 == 0.0
    
    def test_calculate_f1_mixed(self):
        """Test F1 calculation with mixed scores."""
        f1 = calculate_f1(precision=0.8, recall=0.6)
        expected = 2 * (0.8 * 0.6) / (0.8 + 0.6)
        assert abs(f1 - expected) < 0.001
    
    def test_calculate_positives_and_negatives_perfect(self):
        """Test TP/FP/FN calculation with perfect match."""
        gt = ["Juan García", "Dr. Martínez", "12/05/2023"]
        pred = ["Juan García", "Dr. Martínez", "12/05/2023"]
        
        tp, fp, fn = calculate_positives_and_negatives(gt, pred)
        assert tp == 3
        assert fp == 0
        assert fn == 0
    
    def test_calculate_positives_and_negatives_mixed(self):
        """Test TP/FP/FN calculation with mixed results."""
        gt = ["Juan García", "Dr. Martínez", "12/05/2023"]
        pred = ["Juan García", "Dr. López", "Hospital"]
        
        tp, fp, fn = calculate_positives_and_negatives(gt, pred)
        assert tp == 1  # Juan García
        assert fp == 2  # Dr. López, Hospital
        assert fn == 2  # Dr. Martínez, 12/05/2023


@pytest.mark.unit
class TestSimilarityMetrics:
    """Test similarity metrics functionality."""
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical texts."""
        similarity = calculate_cosine_similarity("Hello world", "Hello world")
        assert abs(similarity - 1.0) < 1e-10
    
    def test_cosine_similarity_different(self):
        """Test cosine similarity with different texts."""
        similarity = calculate_cosine_similarity("Hello world", "Goodbye universe")
        assert 0.0 <= similarity <= 1.0
    
    def test_cosine_similarity_empty(self):
        """Test cosine similarity with empty texts."""
        similarity = calculate_cosine_similarity("", "")
        assert similarity == 0.0
    
    def test_levenshtein_distance_identical(self):
        """Test Levenshtein distance with identical texts."""
        distance = calculate_levenshtein_distance("Hello", "Hello")
        assert distance == 0
    
    def test_levenshtein_distance_different(self):
        """Test Levenshtein distance with different texts."""
        distance = calculate_levenshtein_distance("kitten", "sitting")
        assert distance == 3
    
    def test_levenshtein_distance_empty(self):
        """Test Levenshtein distance with empty texts."""
        distance = calculate_levenshtein_distance("", "abc")
        assert distance == 3
    
    def test_inverse_levenshtein_zero_distance(self):
        """Test inverse Levenshtein with zero distance."""
        inv_lev = calculate_inverse_levenshtein(0)
        assert inv_lev == 1.0
    
    def test_inverse_levenshtein_positive_distance(self):
        """Test inverse Levenshtein with positive distance."""
        inv_lev = calculate_inverse_levenshtein(5)
        expected = 1.0 / 5.0  # Should be 1/distance, not 1/(distance+1)
        assert abs(inv_lev - expected) < 1e-10


@pytest.mark.unit
class TestEvaluator:
    """Test evaluator functionality."""
    
    def test_evaluate_identification_perfect(self):
        """Test identification evaluation with perfect results."""
        gt_records = ["Juan García", "Dr. Martínez"]
        pred_records = ["Juan García", "Dr. Martínez"]
        
        metrics = evaluate_identification(gt_records, pred_records)
        
        assert metrics["identification_precision"] == 1.0
        assert metrics["identification_recall"] == 1.0
        assert metrics["identification_f1"] == 1.0
        assert metrics["identification_tp"] == 2
        assert metrics["identification_fp"] == 0
        assert metrics["identification_fn"] == 0
    
    def test_evaluate_identification_mixed(self):
        """Test identification evaluation with mixed results."""
        gt_records = ["Juan García", "Dr. Martínez", "12/05/2023"]
        pred_records = ["Juan García", "Dr. López"]
        
        metrics = evaluate_identification(gt_records, pred_records)
        
        assert metrics["identification_tp"] == 1
        assert metrics["identification_fp"] == 1
        assert metrics["identification_fn"] == 2
        assert metrics["identification_precision"] == 0.5
        assert abs(metrics["identification_recall"] - 1/3) < 0.001
    
    def test_evaluate_label_perfect(self):
        """Test label evaluation with perfect results."""
        gt_texts = ["Patient [**OTROS_SUJETO_ASISTENCIA**] was seen."]
        pred_texts = ["Patient [**OTROS_SUJETO_ASISTENCIA**] was seen."]
        gt_labels = [["OTROS_SUJETO_ASISTENCIA"]]
        pred_labels = [["OTROS_SUJETO_ASISTENCIA"]]
        
        metrics = evaluate_label(gt_texts, pred_texts, gt_labels, pred_labels)
        
        assert metrics["label_precision"] == 1.0
        assert metrics["label_recall"] == 1.0
        assert metrics["label_f1"] == 1.0
        assert metrics["label_cosine_sim"] == 1.0
    
    def test_evaluate_label_mixed(self):
        """Test label evaluation with mixed results."""
        gt_texts = ["Patient [**OTROS_SUJETO_ASISTENCIA**] was seen."]
        pred_texts = ["Patient [**NOMBRE_PERSONAL_SANITARIO**] was seen."]
        gt_labels = [["OTROS_SUJETO_ASISTENCIA"]]
        pred_labels = [["NOMBRE_PERSONAL_SANITARIO"]]
        
        metrics = evaluate_label(gt_texts, pred_texts, gt_labels, pred_labels)
        
        assert metrics["label_precision"] == 0.0
        assert metrics["label_recall"] == 0.0
        assert metrics["label_f1"] == 0.0
        assert metrics["label_cosine_sim"] > 0.5  # Reasonable text similarity


@pytest.mark.unit
class TestExtractors:
    """Test extractors functionality."""
    
    def test_extract_labels_from_masked_text_single(self):
        """Test extracting labels from text with single entity."""
        text = "Patient [**OTROS_SUJETO_ASISTENCIA**] was seen."
        labels = extract_labels_from_masked_text(text)
        assert labels == ["OTROS_SUJETO_ASISTENCIA"]
    
    def test_extract_labels_from_masked_text_multiple(self):
        """Test extracting labels from text with multiple entities."""
        text = "Patient [**OTROS_SUJETO_ASISTENCIA**] was seen by Dr. [**NOMBRE_PERSONAL_SANITARIO**] on [**FECHAS**]."
        labels = extract_labels_from_masked_text(text)
        assert labels == ["OTROS_SUJETO_ASISTENCIA", "NOMBRE_PERSONAL_SANITARIO", "FECHAS"]
    
    def test_extract_labels_from_masked_text_none(self):
        """Test extracting labels from text with no entities."""
        text = "Patient was seen and treated appropriately."
        labels = extract_labels_from_masked_text(text)
        assert labels == []
    
    def test_extract_labels_from_masked_text_empty(self):
        """Test extracting labels from empty text."""
        text = ""
        labels = extract_labels_from_masked_text(text)
        assert labels == []