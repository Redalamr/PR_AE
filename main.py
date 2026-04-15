"""
Pipeline V0 — Point d'entrée principal.

Pipeline complet :
    Touche clavier [S] → Capture image → Prétraitement → Détection blocs
    → Classification (texte/figure) → OCR → Génération PDF

Usage :
    python main.py                          # Mode webcam live
    python main.py --image path/to/img.jpg  # Mode image statique
    python main.py --dir path/to/images/    # Mode batch (dossier)
"""

import cv2
import numpy as np
import time
import logging
import argparse
from pathlib import Path
from typing import List, Optional

import config
from capture.keyboard_trigger import KeyboardCapture
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.enhance import BinarizationMethod
from layout.block_detector import BlockDetector, Block
from layout.heuristic_classifier import HeuristicClassifier
from output.pdf_generator import PDFGenerator

logger = logging.getLogger(__name__)


class WhiteboardPipeline:
    """Pipeline V0 complet : image → PDF."""

    def __init__(
        self, use_cnn: bool = False, use_doctr: bool = False,
        binarization: str = "adaptive_clahe", skip_perspective: bool = False,
    ):
        logger.info("Initialisation du pipeline V0…")

        self.capture = KeyboardCapture()
        self.preprocessing = PreprocessingPipeline(
            binarization_method=BinarizationMethod(binarization),
            skip_perspective=skip_perspective,
        )
        self.block_detector = BlockDetector()

        if use_cnn:
            from layout.cnn_classifier import CNNBlockClassifier
            self.classifier = CNNBlockClassifier()
            logger.info("Classifieur CNN chargé")
        else:
            self.classifier = HeuristicClassifier()
            logger.info("Classifieur heuristique activé")

        if use_doctr:
            from ocr.doctr_engine import DocTREngine
            self.ocr_engine = DocTREngine()
            logger.info("OCR docTR chargé")
        else:
            from ocr.tesseract_engine import TesseractEngine
            self.ocr_engine = TesseractEngine()
            logger.info("OCR Tesseract activé")

        self.pdf_generator = None
        logger.info("Pipeline V0 initialisé ✓")

    def process_image(self, image: np.ndarray, output_pdf: Optional[str] = None) -> Path:
        """Traite une image et génère un PDF."""
        start_time = time.time()
        logger.info(f"Traitement : {image.shape}")

        # 1. Prétraitement
        t0 = time.time()
        binary = self.preprocessing.run(image)
        logger.info(f"  [1/4] Prétraitement : {time.time() - t0:.2f}s")

        # 2. Détection de blocs
        t0 = time.time()
        blocks = self.block_detector.detect(binary)
        logger.info(f"  [2/4] Détection : {len(blocks)} blocs en {time.time() - t0:.2f}s")

        # 3. Classification
        t0 = time.time()
        for block in blocks:
            if block.image is not None:
                label, conf = self.classifier.classify(block.image)
                block.label = label
                block.confidence = conf
        logger.info(f"  [3/4] Classification : {time.time() - t0:.2f}s")

        # 4. OCR + PDF
        t0 = time.time()
        self.pdf_generator = PDFGenerator()
        self.pdf_generator.add_title("Capture du Tableau Blanc")

        for block in blocks:
            if block.label == "text" and block.image is not None:
                ocr_result = self.ocr_engine.recognize(block.image)
                self.pdf_generator.add_text(ocr_result.text)
            elif block.label == "figure" and block.image is not None:
                self.pdf_generator.add_figure(block.image, caption="[Figure]")

        pdf_path = self.pdf_generator.save(output_pdf)
        logger.info(f"  [4/4] OCR + PDF : {time.time() - t0:.2f}s")

        total_time = time.time() - start_time
        logger.info(f"Pipeline terminé en {total_time:.2f}s → {pdf_path}")
        return pdf_path

    def run_live(self):
        """Mode live : capture webcam + traitement."""
        image = self.capture.start_live_capture()
        if image is not None:
            return self.process_image(image)
        logger.warning("Aucune image capturée")

    def run_static(self, image_path: str, output_pdf: Optional[str] = None) -> Path:
        """Mode statique : traitement d'une image fichier."""
        image = self.capture.load_static_image(image_path)
        return self.process_image(image, output_pdf)

    def run_batch(self, image_dir: str) -> List[Path]:
        """Mode batch : traitement de toutes les images d'un dossier."""
        image_dir = Path(image_dir)
        image_files = sorted(
            list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
        )
        logger.info(f"Mode batch : {len(image_files)} images")

        results = []
        for img_file in image_files:
            try:
                pdf_path = self.run_static(str(img_file), f"{img_file.stem}_ocr.pdf")
                results.append(pdf_path)
            except Exception as e:
                logger.error(f"Erreur sur {img_file.name} : {e}")

        logger.info(f"Batch : {len(results)}/{len(image_files)} PDFs")
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline V0 — OCR Tableau Blanc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py                            # Mode webcam live
  python main.py --image photo.jpg          # Image statique
  python main.py --dir images/              # Batch (dossier)
  python main.py --image photo.jpg --cnn    # Avec CNN
  python main.py --image photo.jpg --doctr  # Avec docTR
        """
    )
    parser.add_argument("--image", type=str, help="Image statique")
    parser.add_argument("--dir", type=str, help="Dossier d'images (batch)")
    parser.add_argument("--output", type=str, default=None, help="Nom du PDF")
    parser.add_argument("--cnn", action="store_true", help="Utiliser le CNN")
    parser.add_argument("--doctr", action="store_true", help="Utiliser docTR")
    parser.add_argument("--no-perspective", action="store_true")
    parser.add_argument("--binarization",
                        choices=["otsu", "adaptive", "otsu_clahe", "adaptive_clahe"],
                        default="adaptive_clahe")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline = WhiteboardPipeline(
        use_cnn=args.cnn, use_doctr=args.doctr,
        binarization=args.binarization, skip_perspective=args.no_perspective,
    )

    if args.dir:
        pipeline.run_batch(args.dir)
    elif args.image:
        pipeline.run_static(args.image, args.output)
    else:
        pipeline.run_live()


if __name__ == "__main__":
    main()
