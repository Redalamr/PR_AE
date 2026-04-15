"""
Tests d'intégration — Pipeline complet V0.
Vérifie le pipeline bout en bout sur des images de test synthétiques.
Mesure le temps d'exécution total.
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


def create_test_image(width=800, height=600) -> np.ndarray:
    """Crée une image de test simulant un tableau blanc."""
    image = np.ones((height, width, 3), dtype=np.uint8) * 240
    cv2.rectangle(image, (50, 40), (width - 50, height - 40), (200, 200, 200), 3)

    for i in range(5):
        y = 80 + i * 40
        cv2.line(image, (80, y), (width - 80, y), (30, 30, 30), 1)
        for j in range(8):
            x = 80 + j * 80
            cv2.rectangle(image, (x, y - 8), (x + 50, y + 2), (50, 50, 50), -1)

    cv2.circle(image, (400, 400), 80, (40, 40, 40), 2)
    cv2.arrowedLine(image, (400, 400), (450, 350), (40, 40, 40), 2)
    return image


def test_preprocessing():
    """Teste le module de prétraitement."""
    from preprocessing.pipeline import PreprocessingPipeline

    print("\n=== Test Prétraitement ===")
    image = create_test_image()
    pipeline = PreprocessingPipeline(skip_perspective=True)

    start = time.time()
    result = pipeline.run(image)
    elapsed = time.time() - start

    assert result is not None, "Résultat None"
    assert len(result.shape) == 2, f"Shape incorrect : {result.shape}"
    assert result.dtype == np.uint8, f"dtype incorrect : {result.dtype}"
    print(f"  ✓ Prétraitement OK — {result.shape}, {elapsed:.3f}s")
    return result


def test_block_detection(binary_image: np.ndarray):
    """Teste la détection des blocs."""
    from layout.block_detector import BlockDetector

    print("\n=== Test Détection de blocs ===")
    detector = BlockDetector()

    start = time.time()
    blocks = detector.detect(binary_image)
    elapsed = time.time() - start

    assert isinstance(blocks, list)
    print(f"  ✓ Détection OK — {len(blocks)} blocs, {elapsed:.3f}s")
    for i, block in enumerate(blocks):
        print(f"    [{i}] bbox={block.bbox}, area={block.area}")
    return blocks


def test_heuristic_classifier(blocks):
    """Teste le classifieur heuristique."""
    from layout.heuristic_classifier import HeuristicClassifier

    print("\n=== Test Classifieur Heuristique ===")
    clf = HeuristicClassifier()

    for i, block in enumerate(blocks):
        if block.image is not None:
            label, conf = clf.classify(block.image)
            block.label = label
            block.confidence = conf
            print(f"  [{i}] {label} (conf={conf:.2f})")

    print(f"  ✓ Classification heuristique OK")
    return blocks


def test_pdf_generation(blocks):
    """Teste la génération de PDF."""
    from output.pdf_generator import PDFGenerator

    print("\n=== Test Génération PDF ===")
    gen = PDFGenerator(title="Test Intégration V0")
    gen.add_title("Résultat du Test d'Intégration")
    gen.add_text("Ceci est un test du pipeline complet.")

    for block in blocks:
        if block.label == "text":
            gen.add_text("[Texte reconnu par OCR serait ici]")
        elif block.label == "figure" and block.image is not None:
            gen.add_figure(block.image, caption="Figure détectée")

    output_path = gen.save("test_integration.pdf")
    assert output_path.exists(), f"PDF non généré : {output_path}"
    print(f"  ✓ PDF généré → {output_path}")


def test_full_pipeline():
    """Test d'intégration complet du pipeline V0."""
    print("\n" + "=" * 60)
    print("  TEST D'INTÉGRATION COMPLET — PIPELINE V0")
    print("=" * 60)

    total_start = time.time()

    image = create_test_image()
    print(f"\nImage de test : {image.shape}")

    binary = test_preprocessing()
    blocks = test_block_detection(binary)
    blocks = test_heuristic_classifier(blocks)
    test_pdf_generation(blocks)

    total_elapsed = time.time() - total_start
    target_met = total_elapsed < config.EVAL_PIPELINE_TIME

    print(f"\n{'=' * 60}")
    print(f"  TEMPS TOTAL : {total_elapsed:.2f}s (cible : <{config.EVAL_PIPELINE_TIME}s)")
    print(f"  STATUT : {'✓ PASS' if target_met else '✗ FAIL'}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_full_pipeline()
