"""
Baseline 2 — TrOCR (HuggingFace, pré-entraîné anglais).
Aucun entraînement. Modèle Transformer OCR off-the-shelf.

Stack : transformers (HuggingFace)
"""

import cv2
import numpy as np
import torch
from PIL import Image
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


class TrOCREngine:
    """Moteur OCR TrOCR — baseline DL pré-entraînée."""

    def __init__(
        self, model_name: str = config.TROCR_MODEL_NAME,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        logger.info(f"Chargement TrOCR : {model_name}…")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info(f"TrOCREngine prêt sur {self.device}")

    def recognize(self, block_image: np.ndarray) -> OCRResult:
        """Reconnaît le texte dans une image de bloc."""
        if len(block_image.shape) == 2:
            block_image = cv2.cvtColor(block_image, cv2.COLOR_GRAY2RGB)
        else:
            block_image = cv2.cvtColor(block_image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(block_image)
        pixel_values = self.processor(
            images=pil_image, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values, max_new_tokens=200,
                output_scores=True, return_dict_in_generate=True,
            )

        text = self.processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )[0].strip()

        confidence = 0.0
        if outputs.scores:
            log_probs = []
            for score in outputs.scores:
                probs = torch.softmax(score, dim=-1)
                log_probs.append(probs.max(dim=-1).values.item())
            confidence = sum(log_probs) / max(len(log_probs), 1)

        return OCRResult(text=text, confidence=confidence)

    def recognize_batch(self, images: list) -> list:
        return [self.recognize(img) for img in images]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()
    img = cv2.imread(args.image)
    engine = TrOCREngine()
    result = engine.recognize(img)
    print(f"Texte : {result.text}\nConfidence : {result.confidence:.2%}")
