"""
Étape 3 — Détection des blocs (OpenCV + morphologie adaptative).

Stratégie corrigée :
  1. Dilatation morphologique AVANT findContours pour fusionner
     les lettres d'un même mot/ligne au niveau pixellaire.
  2. findContours sur l'image dilatée → contours de lignes entières.
  3. Extraction des régions sur l'image binarisée originale.
  4. Fusion multi-passes pour les cas résiduels.

Cette approche évite la fragmentation lettre-par-lettre qui ruinait l'OCR.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


@dataclass
class Block:
    """Représente un bloc détecté dans l'image."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    label: str = "unknown"
    confidence: float = 0.0
    image: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def x(self) -> int: return self.bbox[0]
    @property
    def y(self) -> int: return self.bbox[1]
    @property
    def w(self) -> int: return self.bbox[2]
    @property
    def h(self) -> int: return self.bbox[3]
    @property
    def area(self) -> int: return self.bbox[2] * self.bbox[3]
    @property
    def aspect_ratio(self) -> float: return self.bbox[2] / max(self.bbox[3], 1)
    @property
    def center(self) -> Tuple[int, int]: return (self.x + self.w // 2, self.y + self.h // 2)


class BlockDetector:
    """
    Détecte les blocs de texte/figure dans une image binarisée.

    Paramètres clés (tous relatifs à la taille de l'image) :
        h_dilate_ratio  : largeur du noyau de dilatation horizontal, en fraction de la
                          largeur image. Contrôle l'aggressivité de la fusion des mots.
        v_dilate_ratio  : hauteur du noyau vertical. Fusionne les interlignes proches.
        min_area_ratio  : surface minimale d'un bloc, en fraction de la surface totale.
    """

    def __init__(
        self,
        # Paramètres de dilatation morphologique (relatifs à l'image)
        h_dilate_ratio: float = 0.015,  # ↓ était 0.04 → fusionnait tout
        v_dilate_ratio: float = 0.004,  # ↓ était 0.008
        h_dilate_iter: int = 1,  
        v_dilate_iter: int = 1,
        # Filtres de taille (relatifs à l'image)
        min_area_ratio: float = 0.0003, # 0.03% de la surface totale
        min_width_ratio: float = 0.01,  # 1% de la largeur
        min_height_ratio: float = 0.005,# 0.5% de la hauteur
        # Fusion résiduelle post-contours
        merge_dist_y_ratio: float = 0.008,  # ↓ était 0.015
        merge_dist_x_ratio: float = 0.015,  # ↓ était 0.03
        padding: int = 4,
    ):
        self.h_dilate_ratio = h_dilate_ratio
        self.v_dilate_ratio = v_dilate_ratio
        self.h_dilate_iter = h_dilate_iter
        self.v_dilate_iter = v_dilate_iter
        self.min_area_ratio = min_area_ratio
        self.min_width_ratio = min_width_ratio
        self.min_height_ratio = min_height_ratio
        self.merge_dist_y_ratio = merge_dist_y_ratio
        self.merge_dist_x_ratio = merge_dist_x_ratio
        self.padding = padding

    def detect(self, binary_image: np.ndarray) -> List[Block]:
        """
        Détecte les blocs dans une image binarisée.

        Args:
            binary_image: Image binarisée (H, W) uint8. Les pixels d'écriture
                          doivent être BLANCS (255) sur fond NOIR (0).

        Returns:
            Liste de Block triés haut→bas, gauche→droite.
        """
        if binary_image is None or binary_image.size == 0:
            logger.warning("BlockDetector.detect : image vide reçue")
            return []

        img_h, img_w = binary_image.shape[:2]
        img_area = img_h * img_w

        # ── Seuils adaptatifs à la résolution ──
        min_area   = max(200, int(img_area   * self.min_area_ratio))
        min_width  = max(10,  int(img_w      * self.min_width_ratio))
        min_height = max(8,   int(img_h      * self.min_height_ratio))
        merge_dist_y = max(5, int(img_h      * self.merge_dist_y_ratio))
        merge_dist_x = max(10,int(img_w      * self.merge_dist_x_ratio))

        # ── Noyaux de dilatation ──
        kw = max(3, int(img_w * self.h_dilate_ratio))
        kh = max(2, int(img_h * self.v_dilate_ratio))
        # Forcer les dimensions impaires (requis par certains kernels OpenCV)
        kw = kw if kw % 2 == 1 else kw + 1
        kh = kh if kh % 2 == 1 else kh + 1

        logger.debug(
            f"BlockDetector — img={img_w}x{img_h}, "
            f"kernel=({kw}x{kh}), min_area={min_area}, "
            f"merge_y={merge_dist_y}, merge_x={merge_dist_x}"
        )

        # ── Étape 1 : Dilatation morphologique pour fusionner les lettres ──
        # Kernel large horizontal → fusionne les lettres d'un mot et les mots d'une ligne
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        dilated = cv2.dilate(binary_image, h_kernel, iterations=self.h_dilate_iter)

        # ── Étape 2 : findContours sur l'image dilatée ──
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        raw_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h >= min_area and w >= min_width and h >= min_height:
                raw_boxes.append((x, y, w, h))

        logger.info(
            f"Contours bruts : {len(contours)}, "
            f"après filtrage taille : {len(raw_boxes)}"
        )

        if not raw_boxes:
            return []

        # ── Étape 3 : Fusion résiduelle multi-passes ──
        merged = self._merge_boxes_multipass(raw_boxes, merge_dist_y, merge_dist_x)
        logger.info(f"Après fusion multi-passes : {len(merged)} blocs")

        # ── Étape 4 : Construction des Block avec crop sur l'image ORIGINALE ──
        blocks = []
        for (x, y, w, h) in merged:
            x_pad = max(0, x - self.padding)
            y_pad = max(0, y - self.padding)
            x2_pad = min(img_w, x + w + self.padding)
            y2_pad = min(img_h, y + h + self.padding)
            w_pad = x2_pad - x_pad
            h_pad = y2_pad - y_pad

            block_img = binary_image[y_pad:y2_pad, x_pad:x2_pad]
            if block_img.size == 0:
                continue

            blocks.append(Block(
                bbox=(x_pad, y_pad, w_pad, h_pad),
                image=block_img.copy(),  # copy() évite les vues partagées
            ))

        blocks.sort(key=lambda b: (b.y, b.x))
        logger.info(f"Blocs finaux : {len(blocks)}")
        return blocks

    def _merge_boxes_multipass(
        self,
        boxes: List[Tuple[int, int, int, int]],
        merge_dist_y: int,
        merge_dist_x: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Fusion multi-passes jusqu'à convergence.

        Contrairement à l'ancien algorithme mono-passe, cette version
        itère jusqu'à ce qu'aucune fusion supplémentaire ne soit possible,
        ce qui garantit que toutes les boîtes proches sont regroupées
        quelle que soit leur position dans la liste triée.
        """
        if not boxes:
            return []

        changed = True
        while changed:
            changed = False
            boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
            merged: List[Tuple[int, int, int, int]] = []
            used = [False] * len(boxes)

            for i in range(len(boxes)):
                if used[i]:
                    continue
                ax, ay, aw, ah = boxes[i]

                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue
                    bx, by, bw, bh = boxes[j]

                    if self._should_merge(
                        (ax, ay, aw, ah), (bx, by, bw, bh),
                        merge_dist_y, merge_dist_x,
                    ):
                        # Absorber b dans a
                        new_x = min(ax, bx)
                        new_y = min(ay, by)
                        new_x2 = max(ax + aw, bx + bw)
                        new_y2 = max(ay + ah, by + bh)
                        ax, ay = new_x, new_y
                        aw, ah = new_x2 - new_x, new_y2 - new_y
                        used[j] = True
                        changed = True

                merged.append((ax, ay, aw, ah))

            boxes = merged

        return boxes

    @staticmethod
    def _should_merge(
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
        dist_y: int,
        dist_x: int,
    ) -> bool:
        """
        Détermine si deux boîtes doivent être fusionnées.

        Critères :
        - Chevauchement horizontal (ou proximité) ET faible écart vertical
          → même ligne de texte
        - OU chevauchement vertical ET faible écart horizontal
          → même colonne / paragraphe adjacent
        """
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        # Intervalles
        a_x1, a_x2 = ax, ax + aw
        a_y1, a_y2 = ay, ay + ah
        b_x1, b_x2 = bx, bx + bw
        b_y1, b_y2 = by, by + bh

        # Chevauchement/proximité horizontale
        h_overlap = not (a_x2 + dist_x < b_x1 or b_x2 + dist_x < a_x1)
        # Chevauchement/proximité verticale
        v_overlap = not (a_y2 + dist_y < b_y1 or b_y2 + dist_y < a_y1)

        # Même ligne : chevauchement horizontal ET proches verticalement
        gap_y = max(0, b_y1 - a_y2, a_y1 - b_y2)
        gap_x = max(0, b_x1 - a_x2, a_x1 - b_x2)

        same_line = h_overlap and gap_y <= dist_y
        same_col  = v_overlap and gap_x <= dist_x

        return same_line or same_col

    def detect_and_classify(
        self, binary_image: np.ndarray, classifier=None
    ) -> List[Block]:
        blocks = self.detect(binary_image)
        if classifier is not None:
            for block in blocks:
                if block.image is not None:
                    label, conf = classifier.classify(block.image)
                    block.label = label
                    block.confidence = conf
        text_count = sum(1 for b in blocks if b.label == "text")
        fig_count  = sum(1 for b in blocks if b.label == "figure")
        logger.info(f"Classification : {text_count} texte, {fig_count} figure")
        return blocks

    def visualize(self, image: np.ndarray, blocks: List[Block]) -> np.ndarray:
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        colors = {
            "text": (0, 220, 80),
            "figure": (0, 80, 255),
            "unknown": (200, 200, 0),
        }
        for block in blocks:
            x, y, w, h = block.bbox
            color = colors.get(block.label, (200, 200, 0))
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            label_text = f"{block.label} {block.confidence:.2f}"
            cv2.putText(
                vis, label_text, (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )
        return vis
