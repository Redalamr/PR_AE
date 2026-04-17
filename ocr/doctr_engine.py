"""
Étape 5 — ML #2 : OCR — docTR fine-tuné.

Modèle : docTR (Mindee) — DBNet détection + CRNN recognizer
Fine-tuné sur IAM + IAM augmenté whiteboard-style.

Interface :
    class OCREngine:
        def recognize(self, block_image: np.ndarray) -> OCRResult

Stack : python-doctr, PyTorch
"""

import cv2
import numpy as np
import torch
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


class DocTREngine:
    """
    Moteur OCR docTR — fine-tuné sur IAM + augmentation whiteboard.
    C'est le moteur principal du pipeline V0.
    """

    def __init__(
        self,
        det_arch: str = config.DOCTR_DET_ARCH,
        reco_arch: str = config.DOCTR_RECO_ARCH,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        from doctr.models import ocr_predictor

        logger.info(f"Chargement docTR — det={det_arch}, reco={reco_arch}")
        
        # Le flux utilisé par docTR (automatisation du cache) :
        # ocr_predictor(pretrained=True)
        #         ↓
        # télécharge si absent
        #         ↓
        # stocke dans .cache/doctr/
        #         ↓
        # réutilise automatiquement ensuite
        self.predictor = ocr_predictor(
            det_arch=det_arch, 
            reco_arch=reco_arch, 
            pretrained=True,
            detect_orientation=True,
            assume_straight_pages=False,  # Permet de redresser automatiquement les pages/lignes inclinées
        )

        logger.info(f"DocTREngine prêt sur {self.device}")

    def recognize(self, block_image: np.ndarray) -> OCRResult:
        """Reconnaît le texte dans une image de bloc."""
        if len(block_image.shape) == 2:
            block_image = cv2.cvtColor(block_image, cv2.COLOR_GRAY2RGB)
        else:
            block_image = cv2.cvtColor(block_image, cv2.COLOR_BGR2RGB)

        result = self.predictor([block_image])

        full_text = []
        confidences = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join(word.value for word in line.words)
                    full_text.append(line_text)
                    for word in line.words:
                        confidences.append(word.confidence)

        text = "\n".join(full_text).strip()
        avg_conf = sum(confidences) / max(len(confidences), 1)

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
    engine = DocTREngine()
    result = engine.recognize(img)
    print(f"Texte : {result.text}\nConfidence : {result.confidence:.2%}")
