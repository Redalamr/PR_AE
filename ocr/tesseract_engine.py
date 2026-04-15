"""
Baseline 1 — Tesseract 5 (off-the-shelf, mode anglais).
Aucun entraînement. Sert de référence.

Stack : pytesseract
"""

import cv2
import numpy as np
import pytesseract
from dataclasses import dataclass
from typing import Optional
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Résultat d'une reconnaissance OCR."""
    text: str
    confidence: float


class TesseractEngine:
    """Moteur OCR Tesseract 5 — baseline classique."""

    def __init__(
        self, lang: str = config.TESSERACT_LANG,
        tesseract_config: str = config.TESSERACT_CONFIG,
        tesseract_cmd: Optional[str] = None,
    ):
        self.lang = lang
        self.config = tesseract_config
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        logger.info(f"TesseractEngine — lang={lang}")

    def recognize(self, block_image: np.ndarray) -> OCRResult:
        """Reconnaît le texte dans une image de bloc."""
        if len(block_image.shape) == 2:
            block_image = cv2.cvtColor(block_image, cv2.COLOR_GRAY2BGR)

        text = pytesseract.image_to_string(
            block_image, lang=self.lang, config=self.config
        ).strip()

        data = pytesseract.image_to_data(
            block_image, lang=self.lang, config=self.config,
            output_type=pytesseract.Output.DICT,
        )
        confidences = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) > 0]
        avg_conf = sum(confidences) / max(len(confidences), 1) / 100.0

        return OCRResult(text=text, confidence=avg_conf)

    def recognize_batch(self, images: list) -> list:
        return [self.recognize(img) for img in images]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()
    img = cv2.imread(args.image)
    engine = TesseractEngine()
    result = engine.recognize(img)
    print(f"Texte : {result.text}\nConfidence : {result.confidence:.2%}")
