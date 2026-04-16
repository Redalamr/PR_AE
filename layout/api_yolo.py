"""
Module V3 — Détection du tableau blanc via l'Inference API HuggingFace.
"""

import requests
import numpy as np
import logging
import cv2
from dataclasses import dataclass
from typing import Optional
from PIL import Image
import io
import base64

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    found: bool
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float
    label: str
    error_message: Optional[str] = None

def detect_whiteboard(
    image: np.ndarray,
    hf_api_key: Optional[str] = None,
    api_url: str = config.HF_INFERENCE_API_URL,
    candidate_labels: list = None,
    confidence_threshold: float = config.YOLO_CONFIDENCE_THRESHOLD,
    timeout: int = config.API_CALL_TIMEOUT,
) -> DetectionResult:
    
    if candidate_labels is None:
        candidate_labels = config.YOLO_WHITEBOARD_LABELS

    h_orig, w_orig = image.shape[:2]
    ratio_w = 1.0
    ratio_h = 1.0

    # ── Conversion, Redimensionnement et Compression en Base64 ──
    try:
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        pil_image = Image.fromarray(rgb_image.astype(np.uint8))
        
        # Redimensionnement si trop grand (Max 1024px)
        max_dim = 1024
        if pil_image.width > max_dim or pil_image.height > max_dim:
            # Calcul du ratio exact pour corriger les coordonnées à la fin
            ratio = min(max_dim / pil_image.width, max_dim / pil_image.height)
            w_resized = int(pil_image.width * ratio)
            h_resized = int(pil_image.height * ratio)
            
            pil_image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            
            ratio_w = w_orig / w_resized
            ratio_h = h_orig / h_resized
            
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format="JPEG", quality=85)
        image_b64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Erreur de conversion d'image : {e}")
        return _fallback_full_image(image, "Erreur conversion B64")

    # ── Préparation des headers et du payload JSON ──
    headers = {"Content-Type": "application/json"}
    if hf_api_key:
        headers["Authorization"] = f"Bearer {hf_api_key}"

    payload = {
        "inputs": image_b64,
        "parameters": {"candidate_labels": candidate_labels}
    }

    # ── Appel à l'Inference API ──
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        detections = response.json()

    except Exception as e:
        logger.error(f"Erreur API YOLO-World : {e}")
        return _fallback_full_image(image, f"Erreur API : {e}")

    # ── Parsing des résultats ──
    if not detections or not isinstance(detections, list):
        return _fallback_full_image(image, "Réponse API vide ou invalide")

    valid = [d for d in detections if isinstance(d, dict) and d.get("score", 0) >= confidence_threshold and "box" in d]

    if not valid:
        return _fallback_full_image(image, "Aucune détection valide")

    best = max(valid, key=lambda d: d["score"])
    box = best["box"]

    # ── Application du ratio pour retrouver les coordonnées sur l'image géante ──
    xmin = max(0, int(box.get("xmin", 0) * ratio_w))
    ymin = max(0, int(box.get("ymin", 0) * ratio_h))
    xmax = min(w_orig, int(box.get("xmax", w_orig) * ratio_w))
    ymax = min(h_orig, int(box.get("ymax", h_orig) * ratio_h))

    logger.info(f"Tableau détecté — score={best['score']:.3f}, box originale=({xmin},{ymin},{xmax},{ymax})")

    return DetectionResult(
        found=True,
        xmin=xmin, ymin=ymin,
        xmax=xmax, ymax=ymax,
        confidence=best["score"],
        label=best.get("label", "whiteboard"),
    )

def crop_whiteboard(image: np.ndarray, detection: DetectionResult) -> np.ndarray:
    if not detection.found:
        return image.copy()
    cropped = image[detection.ymin:detection.ymax, detection.xmin:detection.xmax]
    if cropped.size == 0:
        return image.copy()
    return cropped

def _fallback_full_image(image: np.ndarray, reason: str) -> DetectionResult:
    h, w = image.shape[:2]
    return DetectionResult(
        found=False, xmin=0, ymin=0, xmax=w, ymax=h,
        confidence=0.0, label="full_image_fallback", error_message=reason,
    )