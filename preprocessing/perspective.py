"""
Étape 2a — Détection des 4 coins du tableau + correction de perspective.

Algorithme :
1. Canny edge detection
2. findContours → filtrage par aire (> 5% de l'image)
3. approxPolyDP → recherche de quadrilatère
4. Tri des 4 coins (haut-gauche, haut-droite, bas-droite, bas-gauche)
5. findHomography + warpPerspective → image redressée

Stack : OpenCV
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


class PerspectiveCorrector:
    """Détecte les 4 coins du tableau et corrige la perspective."""

    def __init__(
        self,
        canny_low: int = config.CANNY_LOW,
        canny_high: int = config.CANNY_HIGH,
        approx_epsilon: float = config.CORNER_APPROX_EPSILON,
        min_area_ratio: float = config.MIN_CONTOUR_AREA_RATIO,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.approx_epsilon = approx_epsilon
        self.min_area_ratio = min_area_ratio

    def detect_board_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Détecte les 4 coins du tableau blanc dans l'image.

        Args:
            image: Image BGR (H, W, 3).

        Returns:
            Array (4, 2) des coins ordonnés [TL, TR, BR, BL] ou None.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Dilater les edges pour fermer les trous
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            logger.warning("Aucun contour trouvé")
            return None

        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * self.min_area_ratio

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.approx_epsilon * peri, True)

            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                ordered = self._order_corners(corners)
                logger.info(f"Tableau détecté — coins : {ordered.tolist()}")
                return ordered

        logger.warning("Aucun quadrilatère trouvé parmi les contours")
        return None

    def correct_perspective(
        self,
        image: np.ndarray,
        corners: Optional[np.ndarray] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Corrige la perspective de l'image en utilisant les 4 coins du tableau.

        Args:
            image: Image BGR (H, W, 3).
            corners: (4, 2) coins ordonnés. Si None, détecte automatiquement.
            output_size: Taille de sortie (W, H). Si None, calculée.

        Returns:
            Image redressée.
        """
        if corners is None:
            corners = self.detect_board_corners(image)
            if corners is None:
                logger.warning("Pas de coins détectés — retour image originale")
                return image

        if output_size is None:
            output_size = self._compute_output_size(corners)

        w, h = output_size
        dst_corners = np.array([
            [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
        ], dtype=np.float32)

        M, _ = cv2.findHomography(corners, dst_corners, cv2.RANSAC, 5.0)
        if M is None:
            logger.warning("Homographie échouée — retour image originale")
            return image

        warped = cv2.warpPerspective(image, M, (w, h))
        logger.info(f"Perspective corrigée — sortie : {w}x{h}")
        return warped

    def manual_corners(
        self, image: np.ndarray, corners: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Fallback : correction avec des coins spécifiés manuellement."""
        assert len(corners) == 4, "Exactement 4 coins requis"
        pts = np.array(corners, dtype=np.float32)
        return self.correct_perspective(image, pts)

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """Ordonne les coins : TL, TR, BR, BL."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

    @staticmethod
    def _compute_output_size(corners: np.ndarray) -> Tuple[int, int]:
        """Calcule la taille de sortie à partir des 4 coins."""
        tl, tr, br, bl = corners
        w1 = np.linalg.norm(tr - tl)
        w2 = np.linalg.norm(br - bl)
        h1 = np.linalg.norm(bl - tl)
        h2 = np.linalg.norm(br - tr)
        return (int(max(w1, w2)), int(max(h1, h2)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Chemin vers l'image")
    args = parser.parse_args()
    img = cv2.imread(args.image)
    corrector = PerspectiveCorrector()
    result = corrector.correct_perspective(img)
    cv2.imwrite("perspective_corrected.png", result)
    print(f"Image corrigée : {result.shape}")
