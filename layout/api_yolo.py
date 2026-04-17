"""
Module V3 — Détection du tableau blanc via l'Inference API HuggingFace (Modèle DETR).
"""

import requests
import numpy as np
import logging
import cv2
from dataclasses import dataclass
from typing import Optional, List
from PIL import Image
import io

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
    candidate_labels: List[str] = None,
    confidence_threshold: float = config.YOLO_CONFIDENCE_THRESHOLD,
    timeout: int = config.API_CALL_TIMEOUT,
) -> DetectionResult:
    """
    Détecte un tableau blanc en utilisant l'API d'inférence Hugging Face avec le modèle DETR.
    Envoie l'image sous forme de bytes bruts pour une meilleure efficacité.
    """
    
    if candidate_labels is None:
        candidate_labels = config.YOLO_WHITEBOARD_LABELS

    h_orig, w_orig = image.shape[:2]
    ratio_w = 1.0
    ratio_h = 1.0

    # ── Conversion et Redimensionnement ──
    try:
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        pil_image = Image.fromarray(rgb_image.astype(np.uint8))
        
        # Redimensionnement si trop grand (Max 1024px) pour l'API
        max_dim = 1024
        if pil_image.width > max_dim or pil_image.height > max_dim:
            ratio = min(max_dim / pil_image.width, max_dim / pil_image.height)
            w_resized = int(pil_image.width * ratio)
            h_resized = int(pil_image.height * ratio)
            
            pil_image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            
            ratio_w = w_orig / w_resized
            ratio_h = h_orig / h_resized
            
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format="JPEG", quality=85)
        image_bytes = img_buffer.getvalue()
    except Exception as e:
        logger.error(f"Erreur de préparation d'image : {e}")
        return _fallback_full_image(image, f"Erreur préparation image : {e}")

    # ── Préparation des headers ──
    # On envoie l'image directement en tant que bytes bruts
    headers = {"Content-Type": "image/jpeg"}
    if hf_api_key:
        headers["Authorization"] = f"Bearer {hf_api_key}"

    # ── Appel à l'Inference API ──
    try:
        logger.info(f"Appel API Hugging Face ({config.YOLO_WORLD_MODEL_ID})...")
        response = requests.post(
            api_url,
            headers=headers,
            data=image_bytes,
            timeout=timeout,
        )
        
        # Gestion spécifique du chargement du modèle (503 Service Unavailable)
        if response.status_code == 503:
            logger.warning("Le modèle Hugging Face est en cours de chargement. Fallback image complète.")
            return _fallback_full_image(image, "Modèle en cours de chargement sur Hugging Face")
            
        response.raise_for_status()
        detections = response.json()

    except Exception as e:
        logger.error(f"Erreur API Hugging Face : {e}")
        return _fallback_full_image(image, f"Erreur API : {e}")

    # ── Parsing des résultats ──
    if not detections or not isinstance(detections, list):
        return _fallback_full_image(image, "Aucune détection retournée par l'API")

    # Filtrage par score et par labels candidats (si spécifiés)
    # Note: DETR utilise des labels COCO. On cherche les plus proches.
    valid = []
    for d in detections:
        if not isinstance(d, dict) or "score" not in d or "box" not in d:
            continue
            
        score = d["score"]
        label = d.get("label", "").lower()
        
        if score >= confidence_threshold:
            # Si on a des labels candidats, on vérifie si le label détecté en fait partie
            if candidate_labels:
                if any(cand.lower() in label for cand in candidate_labels):
                    valid.append(d)
            else:
                valid.append(d)

    if not valid:
        logger.info("Aucun tableau détecté avec un score suffisant. Utilisation de l'image complète.")
        return _fallback_full_image(image, "Aucun tableau détecté")

    # On prend la détection avec le meilleur score
    best = max(valid, key=lambda d: d["score"])
    box = best["box"]

    # ── Application du ratio pour retrouver les coordonnées sur l'image originale ──
    # Le format de box de DETR via Inference API est {'xmin':, 'ymin':, 'xmax':, 'ymax':}
    xmin = max(0, int(box.get("xmin", 0) * ratio_w))
    ymin = max(0, int(box.get("ymin", 0) * ratio_h))
    xmax = min(w_orig, int(box.get("xmax", w_orig) * ratio_w))
    ymax = min(h_orig, int(box.get("ymax", h_orig) * ratio_h))

    logger.info(f"Tableau détecté ({best['label']}) — score={best['score']:.3f}, box=({xmin},{ymin},{xmax},{ymax})")

    return DetectionResult(
        found=True,
        xmin=xmin, ymin=ymin,
        xmax=xmax, ymax=ymax,
        confidence=best["score"],
        label=best.get("label", "detected_object"),
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