"""
Baseline heuristique pour la classification des blocs (texte vs figure).

Critères : ratio d'aspect + densité de pixels sombres + projection horizontale + Hu moments.
Sert de référence pour comparer avec le CNN fine-tuné.
"""

import cv2
import numpy as np
from typing import Tuple
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


class HeuristicClassifier:
    """Classifieur heuristique de blocs : texte vs figure."""

    def __init__(
        self,
        aspect_ratio_threshold: float = config.HEURISTIC_ASPECT_RATIO_THRESHOLD,
        dark_density_threshold: float = config.HEURISTIC_DARK_DENSITY_THRESHOLD,
    ):
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.dark_density_threshold = dark_density_threshold

    def classify(self, block_image: np.ndarray) -> Tuple[str, float]:
        """
        Classifie un bloc en 'text' ou 'figure'.

        Args:
            block_image: Image du bloc (H, W) uint8 binarisée.

        Returns:
            Tuple (label, confidence).
        """
        h, w = block_image.shape[:2]
        if h == 0 or w == 0:
            return ("unknown", 0.0)

        aspect_ratio = w / max(h, 1)
        dark_density = np.count_nonzero(block_image) / (h * w)

        h_proj = np.sum(block_image > 0, axis=1)
        h_proj_std = np.std(h_proj) / max(np.mean(h_proj), 1)

        moments = cv2.moments(block_image)
        hu = cv2.HuMoments(moments).flatten()

        score = 0.0
        if aspect_ratio > self.aspect_ratio_threshold:
            score += 0.3
        if 0.1 < dark_density < 0.6:
            score += 0.25
        if h_proj_std < 0.8:
            score += 0.25
        if abs(hu[0]) < 0.5:
            score += 0.2

        label = "text" if score >= 0.5 else "figure"
        return (label, score)

    def classify_batch(self, block_images: list) -> list:
        """Classifie une liste de blocs."""
        return [self.classify(img) for img in block_images]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    text_block = np.zeros((50, 200), dtype=np.uint8)
    for i in range(0, 50, 10):
        text_block[i:i+3, 10:190] = 255

    figure_block = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(figure_block, (50, 50), 40, 255, 2)

    clf = HeuristicClassifier()
    print(f"Texte : {clf.classify(text_block)}")
    print(f"Figure : {clf.classify(figure_block)}")
