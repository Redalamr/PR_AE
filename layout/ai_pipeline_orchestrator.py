"""
Module V3 — Orchestrateur du Pipeline IA (YOLO-World + Surya).

Ce module est le point d'entrée du nouveau pipeline "IA". Il enchaîne :
  1. Détection YOLO-World (Inference API HF) → coordonnées du tableau
  2. Crop du tableau sur l'image originale
  3. Zonage Surya (Gradio Space) → liste de blocs typés
  4. Crop de chaque bloc + routing vers OCR/LaTeX-OCR/figure
  5. Correction LLM du texte agrégé
  6. Retour d'un objet PipelineResult (identique au pipeline V2)

CONTRAINTE : La signature de retour est PipelineResult (défini dans pipeline_result.py),
garantissant que tout le code d'affichage Streamlit reste inchangé.

Stack : api_yolo, api_surya, ocr engines (existants), llm corrector (existant)
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Callable, List
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def run_ai_pipeline(
    image: np.ndarray,
    hf_api_key: Optional[str],
    ocr_engine_type: str,
    latex_backend: str,
    llm_provider: str,
    llm_api_key: Optional[str],
    enable_llm: bool,
    enable_latex: bool,
    progress_callback: Optional[Callable[[int, str], None]] = None,
):
    """
    Exécute le Pipeline IA V3 complet.

    Args:
        image: Image OpenCV BGR (H, W, 3) du tableau uploadé.
        hf_api_key: Token HuggingFace pour l'Inference API YOLO-World.
        ocr_engine_type: "tesseract", "doctr" ou "trocr".
        latex_backend: "simulate", "pix2tex" ou "nougat".
        llm_provider: "simulate", "openai" ou "anthropic".
        llm_api_key: Clé API LLM si provider != "simulate".
        enable_llm: Activer la correction LLM.
        enable_latex: Activer le moteur LaTeX-OCR pour les blocs maths.
        progress_callback: Fonction optionnelle (progress_int, message_str) → None.

    Returns:
        PipelineResult (importé depuis pipeline_result.py).
    """
    from pipeline_result import PipelineResult

    def _progress(p: int, msg: str):
        logger.info(f"[Pipeline IA] {p}% — {msg}")
        if progress_callback:
            progress_callback(p, msg)

    start_time = time.time()

    # ─────────────────────────────────────────
    # ÉTAPE 1 : Détection YOLO-World
    # ─────────────────────────────────────────
    _progress(10, "🔍 Détection du tableau blanc (YOLO-World)…")

    from layout.api_yolo import detect_whiteboard, crop_whiteboard
    detection = detect_whiteboard(image, hf_api_key=hf_api_key)

    if not detection.found:
        logger.warning(
            f"YOLO-World n'a pas détecté de tableau "
            f"(raison : {detection.error_message}). "
            "Fallback : utilisation de l'image complète."
        )
        _progress(20, "⚠️ Tableau non détecté — traitement de l'image entière…")
    else:
        _progress(20, f"✅ Tableau détecté (confiance : {detection.confidence:.0%})")

    # Crop du tableau (ou image entière si fallback)
    whiteboard_crop = crop_whiteboard(image, detection)

    # ─────────────────────────────────────────
    # ÉTAPE 2 : Zonage Surya
    # ─────────────────────────────────────────
    _progress(30, "🗂️ Analyse du layout (Surya)…")

    from layout.api_surya import analyze_layout, SuryaBlock
    layout_result = analyze_layout(whiteboard_crop)

    if not layout_result.success:
        logger.error(
            f"Échec du zonage Surya : {layout_result.error_message}. "
            "Fallback : traitement OCR de l'image entière."
        )
        _progress(40, "⚠️ Zonage Surya échoué — OCR de l'image entière…")
        return _fallback_full_ocr(
            image=whiteboard_crop,
            ocr_engine_type=ocr_engine_type,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            enable_llm=enable_llm,
            processing_time=time.time() - start_time,
        )

    blocks: List[SuryaBlock] = layout_result.blocks
    _progress(45, f"✅ {len(blocks)} blocs détectés par Surya")

    # ─────────────────────────────────────────
    # ÉTAPE 3 : Chargement des moteurs OCR
    # ─────────────────────────────────────────
    _progress(50, "⚙️ Chargement des moteurs OCR…")

    ocr_engine = _load_ocr_engine(ocr_engine_type)
    latex_engine = _load_latex_engine(latex_backend) if enable_latex else None

    # ─────────────────────────────────────────
    # ÉTAPE 4 : Routing par bloc Surya
    # ─────────────────────────────────────────
    text_parts = []
    latex_formulas = []
    figure_images = []
    ocr_confidences = []

    text_block_count = 0
    math_block_count = 0
    figure_block_count = 0
    unknown_block_count = 0

    total_blocks = len(blocks)

    for idx, block in enumerate(blocks):
        if block.image is None or block.image.size == 0:
            continue

        routing = block.routing_type
        progress_val = 50 + int((idx / max(total_blocks, 1)) * 35)
        _progress(progress_val, f"📄 Traitement bloc {idx+1}/{total_blocks} ({block.label})…")

        # ── Route : Texte → OCR classique ──
        if routing == "text":
            text_block_count += 1
            try:
                ocr_result = ocr_engine.recognize(block.image)
                if ocr_result.text.strip():
                    text_parts.append(ocr_result.text.strip())
                if hasattr(ocr_result, "confidence") and ocr_result.confidence > 0:
                    ocr_confidences.append(ocr_result.confidence)
            except Exception as e:
                logger.error(f"Erreur OCR sur bloc [{block.label}] idx={idx} : {e}")

        # ── Route : Maths → LaTeX-OCR ──
        elif routing == "math" and enable_latex and latex_engine is not None:
            math_block_count += 1
            try:
                latex_result = latex_engine.recognize(block.image)
                if latex_result.text.strip():
                    latex_formulas.append(latex_result.text.strip())
            except Exception as e:
                logger.error(f"Erreur LaTeX-OCR sur bloc [{block.label}] idx={idx} : {e}")

        # ── Route : Figure → Sauvegarde image ──
        elif routing == "figure":
            figure_block_count += 1
            figure_images.append(block.image.copy())

        # ── Route : Inconnu → Tentative OCR (safe fallback) ──
        else:
            unknown_block_count += 1
            logger.debug(f"Bloc '{block.label}' non routé → tentative OCR.")
            try:
                ocr_result = ocr_engine.recognize(block.image)
                if ocr_result.text.strip():
                    text_parts.append(ocr_result.text.strip())
            except Exception:
                pass

    logger.info(
        f"Routing terminé — text={text_block_count}, math={math_block_count}, "
        f"figure={figure_block_count}, unknown={unknown_block_count}"
    )

    # ─────────────────────────────────────────
    # ÉTAPE 5 : Correction LLM
    # ─────────────────────────────────────────
    _progress(87, "🤖 Correction LLM…")

    raw_text = "\n\n".join(text_parts)
    corrections_count = 0

    if enable_llm and raw_text.strip():
        try:
            from llm.corrector import LLMCorrector
            corrector = LLMCorrector(
                provider=llm_provider,
                api_key=llm_api_key or None,
            )
            correction_result = corrector.correct(raw_text)
            corrected_text = correction_result.corrected_text
            corrections_count = correction_result.corrections_count
        except Exception as e:
            logger.error(f"Erreur correction LLM : {e}")
            corrected_text = raw_text
    else:
        corrected_text = raw_text

    # ─────────────────────────────────────────
    # ÉTAPE 6 : Construction du PipelineResult
    # ─────────────────────────────────────────
    _progress(95, "📦 Finalisation…")

    processing_time = time.time() - start_time
    avg_conf = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else 0.0

    # Image "binarisée" pour l'affichage dans Streamlit :
    # Pour le Pipeline IA, on affiche le crop du tableau (converti en niveaux de gris)
    # pour que l'onglet "Image Binarisée" reste cohérent.
    if len(whiteboard_crop.shape) == 3:
        display_binary = cv2.cvtColor(whiteboard_crop, cv2.COLOR_BGR2GRAY)
    else:
        display_binary = whiteboard_crop

    # Construction de faux Block V0 pour la visualisation
    # (le visualiseur existant attend des layout.block_detector.Block)
    from layout.block_detector import Block as V0Block
    v0_blocks = []
    for sb in blocks:
        fake_block = V0Block(
            bbox=(sb.xmin, sb.ymin, sb.xmax - sb.xmin, sb.ymax - sb.ymin),
            label=sb.routing_type if sb.routing_type != "unknown" else "text",
            confidence=1.0,
            image=sb.image,
        )
        v0_blocks.append(fake_block)

    return PipelineResult(
        raw_text=raw_text,
        corrected_text=corrected_text,
        latex_formulas=latex_formulas,
        figure_images=figure_images,
        blocks=v0_blocks,
        processing_time=processing_time,
        block_count=len(blocks),
        text_block_count=text_block_count,
        math_block_count=math_block_count,
        figure_block_count=figure_block_count,
        corrections_count=corrections_count,
        binary_image=display_binary,
        ocr_avg_confidence=avg_conf,
    )


# ─────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────

def _load_ocr_engine(engine_type: str):
    """Charge le moteur OCR demandé (sans cache Streamlit car appelé hors UI)."""
    if engine_type == "doctr":
        from ocr.doctr_engine import DocTREngine
        return DocTREngine()
    elif engine_type == "trocr":
        from ocr.trocr_engine import TrOCREngine
        return TrOCREngine()
    else:
        from ocr.tesseract_engine import TesseractEngine
        return TesseractEngine()


def _load_latex_engine(backend: str):
    """Charge le moteur LaTeX-OCR demandé."""
    from ocr.latex_ocr_engine import LatexOCREngine
    return LatexOCREngine(backend=backend)


def _fallback_full_ocr(
    image: np.ndarray,
    ocr_engine_type: str,
    llm_provider: str,
    llm_api_key: Optional[str],
    enable_llm: bool,
    processing_time: float,
):
    """
    Fallback : si Surya échoue, on passe l'image entière à l'OCR principal.
    Retourne un PipelineResult minimal.
    """
    from pipeline_result import PipelineResult

    ocr_engine = _load_ocr_engine(ocr_engine_type)
    try:
        ocr_result = ocr_engine.recognize(image)
        raw_text = ocr_result.text
    except Exception as e:
        logger.error(f"Erreur OCR fallback : {e}")
        raw_text = f"[Erreur OCR : {e}]"

    corrected_text = raw_text
    if enable_llm and raw_text.strip():
        try:
            from llm.corrector import LLMCorrector
            corrector = LLMCorrector(provider=llm_provider, api_key=llm_api_key or None)
            res = corrector.correct(raw_text)
            corrected_text = res.corrected_text
        except Exception:
            pass

    display_binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    return PipelineResult(
        raw_text=raw_text, corrected_text=corrected_text,
        latex_formulas=[], figure_images=[], blocks=[],
        processing_time=time.time() - processing_time,
        block_count=0, text_block_count=0, math_block_count=0,
        figure_block_count=0, corrections_count=0,
        binary_image=display_binary, ocr_avg_confidence=0.0,
    )
