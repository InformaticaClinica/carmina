"""Modern CARMEN-style evaluation for identify/label/substitute outputs."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Tuple


_MARKER_PATTERN = re.compile(r"\[\*\*(.*?)\*\*\]", re.DOTALL)


@dataclass
class NormalizationConfig:
    partial_match: bool = True
    partial_match_strategy: str = "substring"
    partial_match_threshold: float = 0.6
    partial_match_min_length: int = 3


@dataclass
class MetricsResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    partial: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "partial": self.partial,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def _extract_markers(text: str) -> List[str]:
    if not text:
        return []
    return [m.strip() for m in _MARKER_PATTERN.findall(text) if m and m.strip()]


def _normalize_token(value: str) -> str:
    collapsed = " ".join(value.split()).lower()
    normalized = unicodedata.normalize("NFD", collapsed)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _normalize_entities(entities: List[str]) -> List[str]:
    return [_normalize_token(entity) for entity in entities if entity and entity.strip()]


def _match_entities(
    pred_entities: List[str],
    gt_entities: List[str],
    config: NormalizationConfig,
) -> Tuple[int, int, int, int, List[Tuple[int, int]]]:
    tp = 0
    partial = 0
    gt_matched = [False] * len(gt_entities)
    pred_matched = [False] * len(pred_entities)
    matches: List[Tuple[int, int]] = []

    for i, pred in enumerate(pred_entities):
        for j, gt in enumerate(gt_entities):
            if pred_matched[i] or gt_matched[j]:
                continue
            if pred == gt:
                tp += 1
                pred_matched[i] = True
                gt_matched[j] = True
                matches.append((i, j))
                break

    if config.partial_match:
        for i, pred in enumerate(pred_entities):
            if pred_matched[i] or len(pred) < config.partial_match_min_length:
                continue
            for j, gt in enumerate(gt_entities):
                if gt_matched[j] or len(gt) < config.partial_match_min_length:
                    continue
                if config.partial_match_strategy == "sequence_matcher":
                    ratio = SequenceMatcher(None, pred, gt).ratio()
                    is_hit = ratio >= config.partial_match_threshold
                else:
                    is_hit = pred in gt or gt in pred
                if is_hit:
                    partial += 1
                    pred_matched[i] = True
                    gt_matched[j] = True
                    matches.append((i, j))
                    break

    fp = sum(1 for matched in pred_matched if not matched)
    fn = sum(1 for matched in gt_matched if not matched)
    return tp, fp, fn, partial, matches


def evaluate_legacy_arrays(
    ground_truth_identity_texts: List[str],
    prediction_identity_texts: List[str],
    ground_truth_texts: List[str],
    prediction_texts: List[str],
    filenames: List[str],
    languages: List[str],
    config: NormalizationConfig | None = None,
) -> Dict:
    config = config or NormalizationConfig()
    doc_results = []
    by_language: Dict[str, Dict[str, MetricsResult]] = {}
    global_m1 = MetricsResult()
    global_m2 = MetricsResult()
    global_m3 = MetricsResult()

    for gt_identify, pred_identify, gt_label, pred_label, filename, language in zip(
        ground_truth_identity_texts,
        prediction_identity_texts,
        ground_truth_texts,
        prediction_texts,
        filenames,
        languages,
    ):
        gt_entities = _extract_markers(gt_identify)
        pred_entities = _extract_markers(pred_identify)
        gt_types = [t.upper() for t in _extract_markers(gt_label)]
        pred_types = [t.upper() for t in _extract_markers(pred_label)]

        if len(gt_types) < len(gt_entities):
            gt_types.extend(["UNKNOWN"] * (len(gt_entities) - len(gt_types)))
        else:
            gt_types = gt_types[: len(gt_entities)]

        if len(pred_types) < len(pred_entities):
            pred_types.extend(["UNKNOWN"] * (len(pred_entities) - len(pred_types)))
        else:
            pred_types = pred_types[: len(pred_entities)]

        gt_norm = _normalize_entities(gt_entities)
        pred_norm = _normalize_entities(pred_entities)

        m1_tp, m1_fp, m1_fn, m1_partial, m1_matches = _match_entities(
            pred_norm, gt_norm, config
        )
        m1 = MetricsResult(tp=m1_tp + m1_partial, fp=m1_fp, fn=m1_fn, partial=m1_partial)

        m2_tp = 0
        m2_fp = 0
        for pred_idx, gt_idx in m1_matches:
            gt_t = gt_types[gt_idx] if gt_idx < len(gt_types) else "UNKNOWN"
            pred_t = pred_types[pred_idx] if pred_idx < len(pred_types) else "UNKNOWN"
            if pred_t == "UNKNOWN":
                continue
            if gt_t == pred_t:
                m2_tp += 1
            else:
                m2_fp += 1
        m2 = MetricsResult(tp=m2_tp, fp=m2_fp, fn=m2_fp)

        gt_pairs = list(zip(gt_norm, gt_types))
        pred_pairs = list(zip(pred_norm, pred_types))
        gt_pair_matched = [False] * len(gt_pairs)
        pred_pair_matched = [False] * len(pred_pairs)
        m3_tp = 0

        for i, (pred_text, pred_type) in enumerate(pred_pairs):
            for j, (gt_text, gt_type) in enumerate(gt_pairs):
                if pred_pair_matched[i] or gt_pair_matched[j]:
                    continue
                if pred_text == gt_text and pred_type == gt_type:
                    pred_pair_matched[i] = True
                    gt_pair_matched[j] = True
                    m3_tp += 1
                    break

        if config.partial_match:
            for i, (pred_text, pred_type) in enumerate(pred_pairs):
                if pred_pair_matched[i] or len(pred_text) < config.partial_match_min_length:
                    continue
                for j, (gt_text, gt_type) in enumerate(gt_pairs):
                    if gt_pair_matched[j] or len(gt_text) < config.partial_match_min_length:
                        continue
                    if (pred_text in gt_text or gt_text in pred_text) and pred_type == gt_type:
                        pred_pair_matched[i] = True
                        gt_pair_matched[j] = True
                        m3_tp += 1
                        break

        m3_fp = sum(1 for matched in pred_pair_matched if not matched)
        m3_fn = sum(1 for matched in gt_pair_matched if not matched)
        m3 = MetricsResult(tp=m3_tp, fp=m3_fp, fn=m3_fn)

        for metric, global_metric in ((m1, global_m1), (m2, global_m2), (m3, global_m3)):
            global_metric.tp += metric.tp
            global_metric.fp += metric.fp
            global_metric.fn += metric.fn
            global_metric.partial += metric.partial

        lang_bucket = by_language.setdefault(
            language or "unknown",
            {"m1": MetricsResult(), "m2": MetricsResult(), "m3": MetricsResult()},
        )
        for name, metric in (("m1", m1), ("m2", m2), ("m3", m3)):
            lang_bucket[name].tp += metric.tp
            lang_bucket[name].fp += metric.fp
            lang_bucket[name].fn += metric.fn
            lang_bucket[name].partial += metric.partial

        doc_results.append(
            {
                "id": filename,
                "language": language,
                "m1": m1.to_dict(),
                "m2": m2.to_dict(),
                "m3": m3.to_dict(),
                "gt_entities": gt_entities,
                "pred_entities": pred_entities,
                "gt_types": gt_types,
                "pred_types": pred_types,
            }
        )

    return {
        "global": {
            "m1": global_m1.to_dict(),
            "m2": global_m2.to_dict(),
            "m3": global_m3.to_dict(),
        },
        "by_language": {
            lang: {metric_name: metric.to_dict() for metric_name, metric in metrics.items()}
            for lang, metrics in by_language.items()
        },
        "document_results": doc_results,
    }
