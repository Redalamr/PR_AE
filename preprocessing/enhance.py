"""
Étape 2b — Normalisation photométrique, binarisation et débruitage.

Pipeline :
1. CLAHE adaptatif (Contrast Limited Adaptive Histogram Equalization)
2. Binarisation : Otsu global ou seuillage adaptatif local
3. Débruitage morphologique : ouverture + fermeture

Stack : OpenCV
"""

import cv2
import numpy as np
from enum import Enum
from typing import Tuple
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


class BinarizationMethod(Enum):
    """Méthodes de binarisation disponibles."""
    OTSU = "otsu"
    ADAPTIVE = "adaptive"
    OTSU_CLAHE = "otsu_clahe"
    ADAPTIVE_CLAHE = "adaptive_clahe"


class ImageEnhancer:
    """
    Applique les transformations photométriques et morphologiques.

    Configurations pour l'ablation study :
    - Config 1 : Otsu seul
    - Config 2 : Adaptatif seul
    - Config 3 : CLAHE + Otsu
    - Config 4 : CLAHE + Adaptatif  (défaut)
    """

    def __init__(
        self,
        method: BinarizationMethod = BinarizationMethod.ADAPTIVE_CLAHE,
        clahe_clip: float = config.CLAHE_CLIP_LIMIT,
        clahe_tile: Tuple[int, int] = config.CLAHE_TILE_SIZE,
        otsu_blur: int = config.OTSU_BLUR_KSIZE,
        adaptive_block: int = config.ADAPTIVE_BLOCK_SIZE,
        adaptive_c: int = config.ADAPTIVE_C,
        morph_kernel: Tuple[int, int] = config.MORPH_KERNEL_SIZE,
        morph_open_iter: int = config.MORPH_OPEN_ITER,
        morph_close_iter: int = config.MORPH_CLOSE_ITER,
        subtract_background: bool = False,
    ):
        self.method = method
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self.otsu_blur = otsu_blur
        self.adaptive_block = adaptive_block
        self.adaptive_c = adaptive_c
        self.morph_kernel = morph_kernel
        self.morph_open_iter = morph_open_iter
        self.morph_close_iter = morph_close_iter
        self.subtract_background = subtract_background

        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=self.clahe_tile,
        )

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Pipeline d'amélioration complet.

        Args:
            image: Image en entrée (BGR ou grayscale).

        Returns:
            Image binarisée et débruitée (H, W) uint8.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # ── NOUVEAU : masque pour ignorer les zones trop sombres (fond de salle) ──
        # Les pixels < 40 de luminosité ne peuvent pas être du tableau blanc
        bright_mask = (gray > 40).astype(np.uint8) * 255

        if self.subtract_background:
            # Estimation du fond avec un filtre médian très fort (noyau 51)
            bg = cv2.medianBlur(gray, 51)
            # Soustraction du fond: on calcule la différence pour que le fond clair devienne foncé
            # cv2.subtract sature à 0 (fond clair - objets sombres => objets sombres)
            diff = cv2.subtract(bg, gray)
            # On inverse pour obtenir un fond blanc
            gray = cv2.bitwise_not(diff)

        use_clahe = self.method in (
            BinarizationMethod.OTSU_CLAHE,
            BinarizationMethod.ADAPTIVE_CLAHE,
        )
        if use_clahe:
            gray = self.apply_clahe(gray)

        use_otsu = self.method in (
            BinarizationMethod.OTSU,
            BinarizationMethod.OTSU_CLAHE,
        )
        if use_otsu:
            binary = self.binarize_otsu(gray)
        else:
            binary = self.binarize_adaptive(gray)

        # ── NOUVEAU : appliquer le masque → zones sombres forcées à 0 ──
        binary = cv2.bitwise_and(binary, bright_mask)

        cleaned = self.morphological_clean(binary)
        return cleaned

    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Applique CLAHE."""
        return self._clahe.apply(gray)

    def binarize_otsu(self, gray: np.ndarray) -> np.ndarray:
        """Binarisation par seuillage Otsu global."""
        blurred = cv2.GaussianBlur(gray, (self.otsu_blur, self.otsu_blur), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary

    def binarize_adaptive(self, gray: np.ndarray) -> np.ndarray:
        """Binarisation par seuillage adaptatif local."""
        block_size = self.adaptive_block
        # cv2.adaptiveThreshold exige un entier impair >= 3
        if block_size < 3:
            block_size = 3
        if block_size % 2 == 0:
            block_size += 1
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            self.adaptive_c,
        )
        return binary

    def morphological_clean(self, binary: np.ndarray) -> np.ndarray:
        """Débruitage par opérations morphologiques (ouverture + fermeture)."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel)
        cleaned = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel, iterations=self.morph_open_iter,
        )
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE, kernel, iterations=self.morph_close_iter,
        )
        return cleaned

    @staticmethod
    def all_methods():
        """Retourne toutes les méthodes pour l'ablation study."""
        return list(BinarizationMethod)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Chemin vers l'image")
    parser.add_argument("--method", choices=[m.value for m in BinarizationMethod],
                        default="adaptive_clahe")
    parser.add_argument("--subtract-bg", action="store_true", help="Activer la soustraction du fond")
    args = parser.parse_args()
    img = cv2.imread(args.image)
    enhancer = ImageEnhancer(method=BinarizationMethod(args.method), subtract_background=args.subtract_bg)
    result = enhancer.enhance(img)
    cv2.imwrite("enhanced.png", result)
    print(f"Image améliorée : {result.shape} — méthode : {args.method}")
