from .classification import calculate_precision, calculate_recall, calculate_f1
from .similarity import calculate_cosine_similarity, calculate_levenshtein_distance
from .extractors import extract_labels_from_masked_text
from .recorder import MetricsRecorder
from .timer import measure_time
from .compare_line import (
    extract_all_metrics,
)

# Convenient re-exports
__all__ = [
    'MetricsRecorder', 'measure_time',
    'evaluate_identification', 'evaluate_label',
    'extract_labels_from_masked_text',
    'extract_tags_from_files',
    'extract_all_metrics',
    'get_all_metrics',
    'calculate_metrics'
]