"""
Évaluation OCR — CER (Character Error Rate) et WER (Word Error Rate).
Compare les 3 moteurs OCR sur IAM test set et IAM augmenté.

Stack : jiwer, matplotlib, seaborn
"""

import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import json
import logging

import jiwer
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


class OCREvaluator:
    """Évaluateur de performances OCR — CER et WER."""

    def __init__(self):
        self.results = []

    def evaluate(
        self, predictions: List[str], references: List[str],
        engine_name: str = "unknown", dataset_name: str = "unknown",
    ) -> Dict[str, float]:
        """Évalue les performances OCR."""
        assert len(predictions) == len(references)

        valid_pairs = [
            (pred, ref) for pred, ref in zip(predictions, references) if ref.strip()
        ]
        if not valid_pairs:
            return {"cer": 1.0, "wer": 1.0}

        preds, refs = zip(*valid_pairs)

        cer = jiwer.cer(list(refs), list(preds))
        wer = jiwer.wer(list(refs), list(preds))
        cer_per_image = [jiwer.cer(ref, pred) for pred, ref in zip(preds, refs)]

        result = {
            "engine": engine_name, "dataset": dataset_name,
            "cer": cer, "wer": wer,
            "cer_median": float(np.median(cer_per_image)),
            "cer_std": float(np.std(cer_per_image)),
            "num_samples": len(valid_pairs),
            "cer_target_met": cer < config.EVAL_CER_TARGET,
            "wer_target_met": wer < config.EVAL_WER_TARGET,
        }
        self.results.append(result)
        logger.info(f"[{engine_name}/{dataset_name}] CER={cer:.4f}, WER={wer:.4f}")
        return result

    def compare_engines(self) -> Dict:
        comparison = {}
        for r in self.results:
            comparison[f"{r['engine']}/{r['dataset']}"] = {"cer": r["cer"], "wer": r["wer"]}
        return comparison

    def plot_comparison(self, save_path: Optional[str] = None, show: bool = True):
        """Génère un graphique comparatif CER/WER."""
        if not self.results:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        labels = [f"{r['engine']}\n({r['dataset']})" for r in self.results]
        cers = [r["cer"] for r in self.results]
        wers = [r["wer"] for r in self.results]

        colors_cer = ["#2ecc71" if c < config.EVAL_CER_TARGET else "#e74c3c" for c in cers]
        axes[0].bar(range(len(cers)), cers, color=colors_cer, alpha=0.8)
        axes[0].axhline(y=config.EVAL_CER_TARGET, color="red", linestyle="--", label=f"Cible ({config.EVAL_CER_TARGET})")
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_xticklabels(labels, fontsize=8)
        axes[0].set_ylabel("CER")
        axes[0].set_title("Character Error Rate")
        axes[0].legend()

        colors_wer = ["#2ecc71" if w < config.EVAL_WER_TARGET else "#e74c3c" for w in wers]
        axes[1].bar(range(len(wers)), wers, color=colors_wer, alpha=0.8)
        axes[1].axhline(y=config.EVAL_WER_TARGET, color="red", linestyle="--", label=f"Cible ({config.EVAL_WER_TARGET})")
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels, fontsize=8)
        axes[1].set_ylabel("WER")
        axes[1].set_title("Word Error Rate")
        axes[1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

    def save_results(self, path: str):
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
