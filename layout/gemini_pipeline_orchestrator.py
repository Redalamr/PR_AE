"""
Module V4 — Orchestrateur du Pipeline Tout-en-un (Gemini 2.5).

Point d'entrée pour la solution unifiée Google Gemini 2.5 Flash.
"""
import cv2
import numpy as np
import time
import logging
from typing import Optional, Callable
from PIL import Image

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from pipeline_result import PipelineResult

logger = logging.getLogger(__name__)

def run_gemini_pipeline(
    image: np.ndarray,
    api_key: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> PipelineResult:
    """
    Exécute le pipeline Tout-en-un avec Gemini 2.5 Flash.
    """
    def _progress(p: int, msg: str):
        logger.info(f"[Gemini Pipeline] {p}% — {msg}")
        if progress_callback:
            progress_callback(p, msg)

    start_time = time.time()
    
    _progress(10, "Configuration de l'API Gemini...")
    if not api_key:
        api_key = config.GEMINI_API_KEY_DEFAULT
        
    genai.configure(api_key=api_key)
    
    # Sécurité (optionnel) - baisser les blocages si c'est de l'OCR pur
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    model = genai.GenerativeModel(config.GEMINI_MODEL_ID, safety_settings=safety_settings)

    _progress(30, "Préparation de l'image (format PIL)...")
    if len(image.shape) == 2:
        pil_img = Image.fromarray(image).convert('RGB')
    else:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    _progress(50, "Appel à Gemini 2.5 Flash (Analyse en cours)...")
    
    max_retries = 3
    retry_delay = 3
    
    output_text = ""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                _progress(50 + attempt*5, f"Appel à Gemini 2.5 Flash (Tentative {attempt+1}/{max_retries})...")
            
            response = model.generate_content([config.GEMINI_SYSTEM_PROMPT, pil_img])
            output_text = response.text
            break
        except Exception as e:
            error_str = str(e)
            logger.error(f"Tentative {attempt+1} échouée avec Gemini API: {error_str}")
            
            if "429" in error_str and attempt < max_retries - 1:
                _progress(50 + attempt*5, f"Quota temporaire atteint (429). Attente de {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff (3s, 6s)
            else:
                output_text = f"**Erreur API Gemini (après {attempt+1} tentatives):** {error_str}"
                break

    _progress(95, "Formatage des résultats...")
    
    processing_time = time.time() - start_time
    
    # Fake binary display to keep Streamlit UI happy
    display_binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # PipelineResult 
    return PipelineResult(
        raw_text=output_text,
        corrected_text=output_text,
        latex_formulas=[], 
        figure_images=[], 
        blocks=[], 
        processing_time=processing_time,
        block_count=1,
        text_block_count=1,
        math_block_count=0,
        figure_block_count=0,
        corrections_count=0,
        binary_image=display_binary,
        ocr_avg_confidence=1.0,
    )
