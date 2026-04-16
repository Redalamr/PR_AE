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
        h, w = block_image.shape[:2]
        if h == 0 or w == 0:
            return ("unknown", 0.0)

        # Normaliser en uint8 si nécessaire
        img = block_image if block_image.dtype == np.uint8 else block_image.astype(np.uint8)

        total_pixels = h * w
        # Pixels d'écriture (blancs sur image binaire inversée)
        ink_pixels = int(np.count_nonzero(img))
        ink_density = ink_pixels / max(total_pixels, 1)

        aspect_ratio = w / max(h, 1)

        # Projection horizontale : somme des pixels blancs par ligne
        h_proj = np.sum(img > 0, axis=1).astype(float)
        mean_proj = float(np.mean(h_proj)) if h_proj.size > 0 else 0.0

        # Coefficient de variation des lignes non-vides
        non_zero_lines = h_proj[h_proj > 0]
        cv_lines = (float(np.std(non_zero_lines)) / max(float(np.mean(non_zero_lines)), 1.0)
                    if len(non_zero_lines) > 1 else 1.0)

        # Nombre de "runs" de lignes actives (groupes de lignes avec pixels)
        line_active = (h_proj > (mean_proj * 0.1)).astype(int)
        transitions = int(np.sum(np.diff(line_active) > 0))  # passages 0→1

        score = 0.0

        # Critère 1 : aspect large → favorable au texte
        if aspect_ratio > self.aspect_ratio_threshold:
            score += 0.30

        # Critère 2 : densité dans la plage texte manuscrit (5%–60%)
        if 0.05 < ink_density < 0.60:
            score += 0.25

        # Critère 3 : faible variation entre lignes → texte régulier
        if cv_lines < 1.0:
            score += 0.25

        # Critère 4 : plusieurs runs de lignes actives → plusieurs lignes de texte
        if transitions >= 2:
            score += 0.20

        label = "text" if score >= 0.50 else "figure"
        return (label, round(score, 3))

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
