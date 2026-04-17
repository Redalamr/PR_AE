"""
Isolateur du tableau blanc — détecte la zone blanche et masque le reste.
Doit être appelé APRÈS le prétraitement, AVANT la détection de blocs.
"""
import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def find_whiteboard_mask(
    binary_image: np.ndarray,
    original_bgr: np.ndarray,
    min_board_ratio: float = 0.10,   # le tableau fait au moins 10% de l'image
) -> np.ndarray:
    """
    Retourne un masque (H, W) uint8 : 255 dans la zone tableau, 0 ailleurs.
    Si aucun tableau n'est détecté, retourne un masque plein (image entière).
    """
    h, w = original_bgr.shape[:2]

    # 1. Trouver la zone la plus blanche (fond tableau blanc)
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    
    # Flou fort pour lisser les marques et ne garder que le fond
    blurred = cv2.GaussianBlur(gray, (51, 51), 0)
    
    # Seuil élevé = zones très blanches (le tableau)
    _, white_mask = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
    
    # 2. Morphologie pour remplir les trous (marques du marqueur)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    filled = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    filled = cv2.morphologyEx(filled, cv2.MORPH_DILATE, kernel, iterations=2)

    # 3. Trouver le plus grand contour blanc
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("WhiteboardLocator : aucun contour blanc trouvé — masque plein")
        return np.ones((h, w), dtype=np.uint8) * 255

    # Filtrer par taille minimale
    min_area = h * w * min_board_ratio
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        logger.warning("WhiteboardLocator : contours trop petits — masque plein")
        return np.ones((h, w), dtype=np.uint8) * 255

    # Prendre le plus grand
    best = max(valid, key=cv2.contourArea)
    
    # 4. Bounding rect du tableau (avec padding)
    x, y, bw, bh = cv2.boundingRect(best)
    pad = 20
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    # 5. Créer le masque rectangulaire
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    area_pct = (bw * bh) / (h * w) * 100
    logger.info(f"WhiteboardLocator : tableau détecté [{x1},{y1}→{x2},{y2}] "
                f"({area_pct:.0f}% de l'image)")
    return mask


def isolate_whiteboard(
    binary_image: np.ndarray,
    original_bgr: np.ndarray,
) -> np.ndarray:
    """
    Applique le masque tableau blanc sur l'image binarisée.
    Tout ce qui est hors du tableau devient noir (0) → ignoré par BlockDetector.
    """
    mask = find_whiteboard_mask(binary_image, original_bgr)
    return cv2.bitwise_and(binary_image, mask)
