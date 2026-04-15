"""
Évaluation du classifieur de blocs — F1, Precision, Recall, matrice de confusion, ROC.
Compare CNN fine-tuné vs heuristique baseline.

Stack : scikit-learn, matplotlib, seaborn
"""

import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import json
import logging

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc,
)
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


class ClassifierEvaluator:
    """Évaluateur du classifieur de blocs."""

    def __init__(self):
        self.results = []

    def evaluate(
        self, y_true: List[int], y_pred: List[int],
        y_prob: Optional[List[float]] = None,
        classifier_name: str = "unknown",
    ) -> Dict:
        """Évalue les performances du classifieur."""
        f1 = f1_score(y_true, y_pred, average="weighted")
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=list(config.BLOCK_CLASSES.values()),
            output_dict=True,
        )

        result = {
            "classifier": classifier_name,
            "f1_score": f1, "precision": precision, "recall": recall,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "num_samples": len(y_true),
            "f1_target_met": f1 > config.EVAL_F1_TARGET,
        }

        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            result["roc_auc"] = roc_auc
            result["fpr"] = fpr.tolist()
            result["tpr"] = tpr.tolist()

        self.results.append(result)
        logger.info(f"[{classifier_name}] F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")
        return result

    def plot_confusion_matrix(self, result: Dict, save_path: Optional[str] = None, show: bool = True):
        """Affiche la matrice de confusion."""
        cm = np.array(result["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(config.BLOCK_CLASSES.values()),
            yticklabels=list(config.BLOCK_CLASSES.values()), ax=ax,
        )
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title(f"Matrice de Confusion — {result['classifier']}")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

    def plot_roc(self, results: Optional[List[Dict]] = None, save_path: Optional[str] = None, show: bool = True):
        """Courbes ROC comparatives."""
        results = results or self.results
        results_with_roc = [r for r in results if "roc_auc" in r]
        if not results_with_roc:
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        for r in results_with_roc:
            ax.plot(r["fpr"], r["tpr"], label=f"{r['classifier']} (AUC={r['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("Courbe ROC — Classifieur de blocs")
        ax.legend()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

    def compare(self) -> Dict:
        return {r["classifier"]: {"f1": r["f1_score"], "precision": r["precision"], "recall": r["recall"]} for r in self.results}

    def save_results(self, path: str):
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
