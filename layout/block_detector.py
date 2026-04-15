"""
Étape 3 — Détection des blocs (heuristique OpenCV).

À partir de l'image binarisée, trouve les régions candidates (bounding boxes)
triées haut→bas, gauche→droite.

Stack : OpenCV — projection horizontale + findContours + filtrage par taille
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
    label: str = "unknown"           # 'text' ou 'figure'
    confidence: float = 0.0
    image: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def w(self) -> int:
        return self.bbox[2]

    @property
    def h(self) -> int:
        return self.bbox[3]

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]

    @property
    def aspect_ratio(self) -> float:
        return self.bbox[2] / max(self.bbox[3], 1)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


class BlockDetector:
    """
    Détecte les blocs candidats dans une image binarisée
    en utilisant findContours + filtrage + tri spatial.

    Interface exposée :
        class BlockDetector:
            def detect_and_classify(self, binary_image) -> List[Block]:
    """

    def __init__(
        self,
        min_area: int = config.MIN_BLOCK_AREA,
        min_width: int = config.MIN_BLOCK_WIDTH,
        min_height: int = config.MIN_BLOCK_HEIGHT,
        merge_dist_y: int = config.BLOCK_MERGE_DISTANCE_Y,
        merge_dist_x: int = config.BLOCK_MERGE_DISTANCE_X,
        padding: int = config.BLOCK_PADDING,
    ):
        self.min_area = min_area
        self.min_width = min_width
        self.min_height = min_height
        self.merge_dist_y = merge_dist_y
        self.merge_dist_x = merge_dist_x
        self.padding = padding

    def detect(self, binary_image: np.ndarray) -> List[Block]:
        """
        Détecte les blocs dans une image binarisée.

        Args:
            binary_image: Image binarisée (H, W) uint8.

        Returns:
            Liste de Block triés haut→bas, gauche→droite.
        """
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        raw_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area >= self.min_area and w >= self.min_width and h >= self.min_height:
                raw_boxes.append((x, y, w, h))

        logger.info(f"Contours : {len(contours)}, après filtrage : {len(raw_boxes)}")

        if not raw_boxes:
            return []

        merged = self._merge_boxes(raw_boxes)
        logger.info(f"Après fusion : {len(merged)} blocs")

        blocks = []
        h_img, w_img = binary_image.shape[:2]
        for (x, y, w, h) in merged:
            x_pad = max(0, x - self.padding)
            y_pad = max(0, y - self.padding)
            w_pad = min(w_img - x_pad, w + 2 * self.padding)
            h_pad = min(h_img - y_pad, h + 2 * self.padding)

            block_img = binary_image[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
            blocks.append(Block(
                bbox=(x_pad, y_pad, w_pad, h_pad),
                image=block_img,
            ))

        blocks.sort(key=lambda b: (b.y, b.x))
        return blocks

    def _merge_boxes(
        self, boxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Fusionne les bounding boxes proches."""
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        merged = [boxes[0]]

        for box in boxes[1:]:
            last = merged[-1]
            lx, ly, lw, lh = last
            bx, by, bw, bh = box

            vertical_close = abs(by - (ly + lh)) < self.merge_dist_y
            horizontal_overlap = (
                bx < lx + lw + self.merge_dist_x and
                bx + bw > lx - self.merge_dist_x
            )

            if vertical_close and horizontal_overlap:
                new_x = min(lx, bx)
                new_y = min(ly, by)
                new_w = max(lx + lw, bx + bw) - new_x
                new_h = max(ly + lh, by + bh) - new_y
                merged[-1] = (new_x, new_y, new_w, new_h)
            else:
                merged.append(box)

        return merged

    def detect_and_classify(
        self, binary_image: np.ndarray, classifier=None
    ) -> List[Block]:
        """
        Détecte les blocs puis les classifie (texte vs figure).

        Args:
            binary_image: Image binarisée.
            classifier: Objet avec méthode classify(block_image) -> (label, confidence).
        """
        blocks = self.detect(binary_image)

        if classifier is not None:
            for block in blocks:
                if block.image is not None:
                    label, conf = classifier.classify(block.image)
                    block.label = label
                    block.confidence = conf

        text_count = sum(1 for b in blocks if b.label == "text")
        fig_count = sum(1 for b in blocks if b.label == "figure")
        logger.info(f"Classification : {text_count} texte, {fig_count} figure")
        return blocks

    def visualize(self, image: np.ndarray, blocks: List[Block]) -> np.ndarray:
        """Dessine les blocs détectés sur l'image pour debug."""
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        colors = {"text": (0, 255, 0), "figure": (0, 0, 255), "unknown": (255, 255, 0)}
        for block in blocks:
            x, y, w, h = block.bbox
            color = colors.get(block.label, (255, 255, 0))
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            label_text = f"{block.label} ({block.confidence:.2f})"
            cv2.putText(vis, label_text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return vis


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Image binarisée")
    args = parser.parse_args()
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    detector = BlockDetector()
    blocks = detector.detect(img)
    print(f"Blocs détectés : {len(blocks)}")
    for i, b in enumerate(blocks):
        print(f"  [{i}] bbox={b.bbox}, area={b.area}, ratio={b.aspect_ratio:.2f}")
