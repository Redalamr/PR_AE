"""
Génération de dataset synthétique pour le classifieur de blocs.

Génère des patches simulant :
- Du texte : lignes horizontales, densité élevée, structure régulière
- Des figures : cercles, courbes, flèches, graphes simples

Stack : OpenCV
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
import random
import logging

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Génère des patches synthétiques de texte et figures."""

    def __init__(self, patch_size: Tuple[int, int] = (224, 224), seed: int = 42):
        self.patch_size = patch_size
        random.seed(seed)
        np.random.seed(seed)

    def generate_text_patch(self) -> np.ndarray:
        """Génère un patch simulant du texte manuscrit."""
        h, w = self.patch_size
        patch = np.ones((h, w), dtype=np.uint8) * 255

        num_lines = random.randint(5, 12)
        line_spacing = h // (num_lines + 1)

        for i in range(num_lines):
            y = (i + 1) * line_spacing + random.randint(-3, 3)
            x_start = random.randint(10, 30)
            x_end = w - random.randint(10, 40)
            num_segments = random.randint(3, 7)
            seg_width = (x_end - x_start) // num_segments

            for j in range(num_segments):
                sx = x_start + j * seg_width + random.randint(0, 5)
                ex = sx + seg_width - random.randint(5, 15)
                if ex <= sx:
                    continue
                thickness = random.randint(1, 3)
                y_var = y + random.randint(-2, 2)
                pts = [[x, y_var + random.randint(-1, 1)] for x in range(sx, ex, 3)]
                if len(pts) > 1:
                    cv2.polylines(patch, [np.array(pts, dtype=np.int32)], False, 0, thickness)

        noise = np.random.normal(0, 5, patch.shape).astype(np.int16)
        return np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def generate_figure_patch(self) -> np.ndarray:
        """Génère un patch simulant une figure."""
        h, w = self.patch_size
        patch = np.ones((h, w), dtype=np.uint8) * 255
        fig_type = random.choice(["circle", "arrow", "graph", "box", "curve"])

        if fig_type == "circle":
            cx, cy = w // 2 + random.randint(-30, 30), h // 2 + random.randint(-30, 30)
            radius = random.randint(30, min(h, w) // 3)
            cv2.circle(patch, (cx, cy), radius, 0, random.randint(1, 3))
            for _ in range(random.randint(0, 3)):
                angle = random.uniform(0, 2 * np.pi)
                x2 = int(cx + radius * 0.7 * np.cos(angle))
                y2 = int(cy + radius * 0.7 * np.sin(angle))
                cv2.line(patch, (cx, cy), (x2, y2), 0, 1)

        elif fig_type == "arrow":
            x1, y1 = random.randint(20, w // 3), random.randint(h // 3, 2 * h // 3)
            x2, y2 = random.randint(2 * w // 3, w - 20), random.randint(h // 3, 2 * h // 3)
            cv2.arrowedLine(patch, (x1, y1), (x2, y2), 0, 2, tipLength=0.1)

        elif fig_type == "graph":
            cv2.line(patch, (30, h - 30), (30, 30), 0, 2)
            cv2.line(patch, (30, h - 30), (w - 30, h - 30), 0, 2)
            pts = []
            for x in range(35, w - 30, 5):
                y = int((h - 60) * (1 - np.sin((x - 35) / (w - 65) * np.pi)) + 30)
                pts.append([x, y + random.randint(-5, 5)])
            if len(pts) > 1:
                cv2.polylines(patch, [np.array(pts)], False, 0, 2)

        elif fig_type == "box":
            num_boxes = random.randint(2, 4)
            boxes = []
            for i in range(num_boxes):
                bx = 30 + i * (w - 60) // num_boxes
                by = random.randint(30, h - 80)
                bw = (w - 60) // num_boxes - 20
                bh = random.randint(30, 60)
                cv2.rectangle(patch, (bx, by), (bx + bw, by + bh), 0, 2)
                boxes.append((bx + bw // 2, by + bh // 2, bx + bw, by + bh))
            for i in range(len(boxes) - 1):
                cv2.line(patch, (boxes[i][2], boxes[i][1]),
                         (boxes[i + 1][0] - 20, boxes[i + 1][1]), 0, 1)

        elif fig_type == "curve":
            pts = np.array([
                [random.randint(20, w // 3), random.randint(20, h - 20)],
                [random.randint(w // 3, 2 * w // 3), random.randint(20, h - 20)],
                [random.randint(2 * w // 3, w - 20), random.randint(20, h - 20)],
            ], dtype=np.int32)
            cv2.polylines(patch, [pts], False, 0, 2)

        noise = np.random.normal(0, 5, patch.shape).astype(np.int16)
        return np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def generate_dataset(self, output_dir: str, num_text: int = 500, num_figure: int = 500):
        """Génère un dataset synthétique complet."""
        output_path = Path(output_dir)
        text_dir = output_path / "text"
        figure_dir = output_path / "figure"
        text_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Génération de {num_text} texte + {num_figure} figure")

        for i in range(num_text):
            cv2.imwrite(str(text_dir / f"text_{i:04d}.png"), self.generate_text_patch())
        for i in range(num_figure):
            cv2.imwrite(str(figure_dir / f"figure_{i:04d}.png"), self.generate_figure_patch())

        logger.info(f"Dataset synthétique → {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-text", type=int, default=500)
    parser.add_argument("--num-figure", type=int, default=500)
    args = parser.parse_args()
    gen = SyntheticDataGenerator()
    gen.generate_dataset(args.output, args.num_text, args.num_figure)
