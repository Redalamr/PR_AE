"""
Module V2 — LaTeX-OCR pour les formules mathématiques.

Quand le classifieur de layout détecte un bloc "equation" ou "math",
ce moteur remplace Tesseract/TrOCR et renvoie la formule au format LaTeX.

Modèles supportés :
  - pix2tex (LaTeX-OCR) — léger, rapide, idéal pour des formules isolées
  - Nougat (Meta) — plus lourd, gère des pages entières de maths

Architecture :
  ┌─────────────┐
  │ BlockDetector│
  └──────┬──────┘
         │ bloc.label == "equation"
         ▼
  ┌──────────────┐     ┌──────────────────┐
  │ LatexOCREngine│────→│ pix2tex / Nougat │
  └──────┬───────┘     └──────────────────┘
         │
         ▼
  OCRResult(text="$\\\\frac{dy}{dx} = ...$", confidence=0.95)

Usage :
    from ocr.latex_ocr_engine import LatexOCREngine
    engine = LatexOCREngine(backend="pix2tex")
    result = engine.recognize(equation_image)
    print(result.text)  # → "\\frac{\\partial L}{\\partial w}"
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Literal

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Résultat d'une reconnaissance OCR (compatible avec les autres engines)."""
    text: str
    confidence: float
    is_latex: bool = False


class LatexOCREngine:
    """
    Moteur OCR spécialisé pour les formules mathématiques.

    Backends :
        - "pix2tex"  : LaTeX-OCR (Lukas Blecher) — rapide, formules isolées
        - "nougat"   : Nougat (Meta) — lourd, pages complètes avec maths
        - "simulate" : Mode simulation — renvoie un placeholder LaTeX

    Args:
        backend: Le backend à utiliser.
        device: "cuda" ou "cpu". Auto-détecté si None.
        model_path: Chemin vers les poids custom (optionnel).

    Branchement dans le pipeline V0 existant :
        Dans main.py ou app.py, après la classification des blocs :

        >>> for block in blocks:
        ...     if block.label in ("equation", "math"):
        ...         result = latex_engine.recognize(block.image)
        ...     elif block.label == "text":
        ...         result = ocr_engine.recognize(block.image)
    """

    def __init__(
        self,
        backend: Literal["pix2tex", "nougat", "simulate"] = "pix2tex",
        device: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        self.backend = backend
        self.device = device
        self.model_path = model_path
        self._model = None

        if backend == "pix2tex":
            self._init_pix2tex()
        elif backend == "nougat":
            self._init_nougat()
        else:
            logger.info("LatexOCREngine — mode simulation activé")

    # ────────────────────────────────────────
    # Initialisation pix2tex
    # ────────────────────────────────────────
    def _init_pix2tex(self):
        """Charge le modèle pix2tex (LaTeX-OCR)."""
        try:
            from pix2tex.cli import LatexOCR

            logger.info("Chargement pix2tex (LaTeX-OCR)…")
            self._model = LatexOCR()
            logger.info("pix2tex chargé ✓")

        except ImportError:
            logger.error(
                "pix2tex non installé. Installez avec : pip install pix2tex\n"
                "Basculement en mode simulation."
            )
            self.backend = "simulate"
        except Exception as e:
            logger.error(f"Erreur lors du chargement de pix2tex : {e}")
            self.backend = "simulate"

    # ────────────────────────────────────────
    # Initialisation Nougat
    # ────────────────────────────────────────
    def _init_nougat(self):
        """Charge le modèle Nougat (Meta)."""
        try:
            import torch
            from transformers import NougatProcessor, VisionEncoderDecoderModel

            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model_name = self.model_path or "facebook/nougat-base"

            logger.info(f"Chargement Nougat : {model_name}…")
            self._processor = NougatProcessor.from_pretrained(model_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self._model.to(self.device)
            self._model.eval()
            logger.info(f"Nougat chargé ✓ sur {self.device}")

        except ImportError:
            logger.error(
                "Nougat / transformers non installé.\n"
                "pip install nougat-ocr transformers torch\n"
                "Basculement en mode simulation."
            )
            self.backend = "simulate"
        except Exception as e:
            logger.error(f"Erreur lors du chargement de Nougat : {e}")
            self.backend = "simulate"

    # ────────────────────────────────────────
    # Reconnaissance
    # ────────────────────────────────────────
    def recognize(self, block_image: np.ndarray) -> OCRResult:
        """
        Reconnaît une formule mathématique et renvoie du LaTeX.

        Args:
            block_image: Image du bloc (H, W) ou (H, W, 3) uint8.

        Returns:
            OCRResult avec text contenant la formule LaTeX.
        """
        if self.backend == "pix2tex":
            return self._recognize_pix2tex(block_image)
        elif self.backend == "nougat":
            return self._recognize_nougat(block_image)
        else:
            return self._simulate_recognition(block_image)

    def _recognize_pix2tex(self, block_image: np.ndarray) -> OCRResult:
        """Reconnaissance via pix2tex."""
        from PIL import Image

        # Conversion numpy → PIL
        if len(block_image.shape) == 2:
            pil_image = Image.fromarray(block_image).convert("RGB")
        else:
            rgb = cv2.cvtColor(block_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

        try:
            latex_str = self._model(pil_image)
            # Nettoyage du résultat
            latex_str = self._clean_latex(latex_str)

            logger.info(f"pix2tex → {latex_str[:80]}…")
            return OCRResult(
                text=latex_str,
                confidence=0.90,  # pix2tex ne fournit pas de score de confiance natif
                is_latex=True,
            )
        except Exception as e:
            logger.error(f"Erreur pix2tex : {e}")
            return OCRResult(text="[Erreur LaTeX-OCR]", confidence=0.0, is_latex=True)

    def _recognize_nougat(self, block_image: np.ndarray) -> OCRResult:
        """Reconnaissance via Nougat (Meta)."""
        import torch
        from PIL import Image

        if len(block_image.shape) == 2:
            pil_image = Image.fromarray(block_image).convert("RGB")
        else:
            rgb = cv2.cvtColor(block_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

        try:
            pixel_values = self._processor(
                images=pil_image, return_tensors="pt"
            ).pixel_values.to(self.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    pixel_values,
                    max_new_tokens=512,
                    bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                )

            decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
            decoded = self._processor.post_process_generation(decoded, fix_markdown=False)

            latex_str = self._clean_latex(decoded)
            logger.info(f"Nougat → {latex_str[:80]}…")

            return OCRResult(text=latex_str, confidence=0.85, is_latex=True)

        except Exception as e:
            logger.error(f"Erreur Nougat : {e}")
            return OCRResult(text="[Erreur Nougat-OCR]", confidence=0.0, is_latex=True)

    def _simulate_recognition(self, block_image: np.ndarray) -> OCRResult:
        """
        Mode simulation — renvoie un placeholder LaTeX crédible.
        Utile pour les tests sans modèle chargé.
        """
        h, w = block_image.shape[:2]
        aspect = w / max(h, 1)

        # Heuristique simple pour simuler des formules réalistes
        if aspect > 3:
            latex = r"\frac{\partial \mathcal{L}}{\partial w} = -\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i) \cdot x_i"
        elif aspect > 1.5:
            latex = r"\sigma(z) = \frac{1}{1 + e^{-z}}"
        else:
            latex = r"\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot R\right]"

        logger.info(f"Simulation LaTeX → {latex[:50]}…")
        return OCRResult(text=latex, confidence=0.95, is_latex=True)

    # ────────────────────────────────────────
    # Utilitaires
    # ────────────────────────────────────────
    @staticmethod
    def _clean_latex(raw: str) -> str:
        """Nettoie la sortie LaTeX brute."""
        # Supprime les espaces en trop
        cleaned = raw.strip()
        # Supprime les balises \[ \] si wrapping inutile sur une formule isolée
        if cleaned.startswith("\\[") and cleaned.endswith("\\]"):
            cleaned = cleaned[2:-2].strip()
        if cleaned.startswith("$$") and cleaned.endswith("$$"):
            cleaned = cleaned[2:-2].strip()
        return cleaned

    def recognize_batch(self, images: list) -> list:
        """Reconnaît un batch de formules."""
        return [self.recognize(img) for img in images]


# ────────────────────────────────────────────
# Fonction utilitaire pour le routage dans le pipeline
# ────────────────────────────────────────────
def is_math_block(label: str) -> bool:
    """
    Détermine si un label de bloc correspond à une formule mathématique.

    Usage dans le pipeline :
        >>> if is_math_block(block.label):
        ...     result = latex_engine.recognize(block.image)
        ... else:
        ...     result = text_engine.recognize(block.image)
    """
    math_labels = {"equation", "math", "formula", "latex", "mathematical"}
    return label.lower().strip() in math_labels


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test en mode simulation
    engine = LatexOCREngine(backend="simulate")

    # Formule large (ratio > 3)
    wide = np.zeros((50, 300), dtype=np.uint8)
    result = engine.recognize(wide)
    print(f"Wide  → {result.text}")
    print(f"         is_latex={result.is_latex}, conf={result.confidence:.2f}")

    # Formule carrée
    square = np.zeros((100, 100), dtype=np.uint8)
    result = engine.recognize(square)
    print(f"Square → {result.text}")
    print(f"          is_latex={result.is_latex}, conf={result.confidence:.2f}")
