"""
Étape 2 — Prétraitement complet (façade unique).

Interface exposée (contrat pour les autres modules) :
    class PreprocessingPipeline:
        def run(self, raw_frame: np.ndarray) -> np.ndarray:
            # retourne image binarisée redressée (H, W) uint8
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional
from pathlib import Path

from .perspective import PerspectiveCorrector
from .enhance import ImageEnhancer, BinarizationMethod

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Façade unique du module de prétraitement.
    Appelée par le pipeline principal (main.py).
    """

    def __init__(
        self,
        binarization_method: BinarizationMethod = BinarizationMethod.ADAPTIVE_CLAHE,
        skip_perspective: bool = False,
        subtract_background: bool = False,
    ):
        self.corrector = PerspectiveCorrector()
        self.enhancer = ImageEnhancer(method=binarization_method, subtract_background=subtract_background)
        self.skip_perspective = skip_perspective
        logger.info(
            f"PreprocessingPipeline — binarisation={binarization_method.value}, "
            f"skip_perspective={skip_perspective}"
        )

    def run(self, raw_frame: np.ndarray) -> np.ndarray:
        """
        Pipeline de prétraitement complet.

        Args:
            raw_frame: Image brute BGR (H, W, 3).

        Returns:
            Image binarisée redressée (H, W) uint8.
        """
        start_time = time.time()

        if not self.skip_perspective:
            corrected = self.corrector.correct_perspective(raw_frame)
        else:
            corrected = raw_frame

        binary = self.enhancer.enhance(corrected)

        elapsed = time.time() - start_time
        logger.info(f"Prétraitement terminé en {elapsed:.2f}s — sortie : {binary.shape}")
        return binary

    def run_with_intermediates(self, raw_frame: np.ndarray) -> dict:
        """
        Pipeline complet retournant les résultats intermédiaires.
        Utile pour le debug et l'ablation study.
        """
        start = time.time()
        results = {"original": raw_frame.copy()}

        if not self.skip_perspective:
            corrected = self.corrector.correct_perspective(raw_frame)
        else:
            corrected = raw_frame
        results["corrected"] = corrected.copy()

        if len(corrected.shape) == 3:
            gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        else:
            gray = corrected.copy()
        results["gray"] = gray.copy()

        clahe = self.enhancer.apply_clahe(gray)
        results["clahe"] = clahe.copy()

        binary = self.enhancer.binarize_adaptive(clahe)
        results["binary"] = binary.copy()

        cleaned = self.enhancer.morphological_clean(binary)
        results["cleaned"] = cleaned.copy()

        results["elapsed_ms"] = (time.time() - start) * 1000
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Image à prétraiter")
    parser.add_argument("--no-perspective", action="store_true")
    parser.add_argument("--method", choices=[m.value for m in BinarizationMethod],
                        default="adaptive_clahe")
    parser.add_argument("--subtract-bg", action="store_true", help="Activer la soustraction du fond")
    args = parser.parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    pipeline = PreprocessingPipeline(
        binarization_method=BinarizationMethod(args.method),
        skip_perspective=args.no_perspective,
        subtract_background=args.subtract_bg,
    )
    result = pipeline.run(img)
    cv2.imwrite("preprocessed.png", result)
    print(f"Image prétraitée : {result.shape}")
