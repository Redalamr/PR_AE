"""
Module V3 — Zonage du tableau recadré via Surya (Gradio Space).

Envoie l'image recadrée du tableau à l'API Gradio du Space "xiaoyao9184/surya"
et retourne une liste de blocs avec leur type (Text, Math, Figure, etc.)
et leurs coordonnées bounding box.

Endpoint utilisé : /layout_det_img
Retour brut : tuple (gallery_images, json_dict)
  → On utilise result[1] qui contient le JSON des détections.

Structure attendue de result[1] :
    {
      "results": [
        {"bbox": [x_min, y_min, x_max, y_max], "label": "Text"},
        {"bbox": [x_min, y_min, x_max, y_max], "label": "Formula"},
        ...
      ]
    }
    OU selon la version du Space :
    [{"bbox": [...], "label": "..."}, ...]

Stack : gradio_client, numpy, PIL, tempfile
"""

import tempfile
import numpy as np
import logging
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


@dataclass
class SuryaBlock:
    """Représente un bloc détecté par Surya."""
    bbox: List[int]          # [xmin, ymin, xmax, ymax]
    label: str               # "Text", "Formula", "Figure", etc.
    image: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def xmin(self) -> int: return self.bbox[0]
    @property
    def ymin(self) -> int: return self.bbox[1]
    @property
    def xmax(self) -> int: return self.bbox[2]
    @property
    def ymax(self) -> int: return self.bbox[3]

    @property
    def area(self) -> int:
        return max(0, self.xmax - self.xmin) * max(0, self.ymax - self.ymin)

    @property
    def routing_type(self) -> str:
        """Renvoie le type de routing normalisé : 'text', 'math', 'figure', 'unknown'."""
        label_lower = self.label.lower()
        if any(t in label_lower for t in [lbl.lower() for lbl in config.SURYA_TEXT_LABELS]):
            return "text"
        if any(t in label_lower for t in [lbl.lower() for lbl in config.SURYA_MATH_LABELS]):
            return "math"
        if any(t in label_lower for t in [lbl.lower() for lbl in config.SURYA_FIGURE_LABELS]):
            return "figure"
        return "unknown"


@dataclass
class SuryaLayoutResult:
    """Résultat complet du zonage Surya."""
    blocks: List[SuryaBlock]
    raw_json: Any
    error_message: Optional[str] = None

    @property
    def success(self) -> bool:
        return len(self.blocks) > 0 and self.error_message is None


def analyze_layout(
    image: np.ndarray,
    space_id: str = config.SURYA_SPACE_ID,
    api_name: str = config.SURYA_API_NAME,
    timeout: int = config.API_CALL_TIMEOUT,
) -> SuryaLayoutResult:
    """
    Envoie l'image recadrée à Surya et retourne les blocs de layout.

    Args:
        image: Image recadrée du tableau (BGR ou RGB), numpy uint8.
        space_id: Identifiant du Space Gradio Surya.
        api_name: Nom de l'endpoint Gradio.
        timeout: Timeout en secondes.

    Returns:
        SuryaLayoutResult avec la liste des SuryaBlock détectés.
    """
    try:
        from gradio_client import Client, handle_file
    except ImportError:
        logger.error(
            "gradio_client non installé. "
            "Installez avec : pip install gradio_client>=0.16.0"
        )
        return SuryaLayoutResult(
            blocks=[], raw_json=None,
            error_message="gradio_client non installé",
        )

    # ── Sauvegarde temporaire de l'image pour handle_file ──
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ) as tmp:
            tmp_path = tmp.name

        # Conversion BGR → RGB si nécessaire pour PIL/Surya
        if len(image.shape) == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img_rgb.astype(np.uint8))
        pil_img.save(tmp_path, format="PNG")

        logger.info(f"Image temporaire sauvegardée : {tmp_path} ({image.shape})")

    except Exception as e:
        logger.error(f"Erreur de sauvegarde temporaire : {e}")
        return SuryaLayoutResult(
            blocks=[], raw_json=None,
            error_message=f"Erreur création fichier temp : {e}",
        )

    # ── Appel Gradio ──
    try:
        logger.info(f"Connexion au Space Gradio : {space_id}")
        client = Client(space_id)

        logger.info(f"Appel endpoint {api_name}…")
        result = client.predict(
            pil_image=handle_file(tmp_path),
            api_name=api_name,
        )

        # result est un tuple : (gallery_data, json_dict)
        raw_json = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else result
        logger.info(f"Réponse Surya reçue. Type : {type(raw_json)}")

    except Exception as e:
        logger.error(f"Erreur lors de l'appel Gradio Surya : {e}")
        return SuryaLayoutResult(
            blocks=[], raw_json=None,
            error_message=f"Erreur appel Gradio : {e}",
        )
    finally:
        # Nettoyage du fichier temporaire
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    # ── Parsing du JSON ──
    blocks = _parse_surya_json(raw_json, image)
    logger.info(f"Surya : {len(blocks)} blocs détectés.")
    for b in blocks:
        logger.debug(f"  → [{b.label}] bbox={b.bbox}, routing={b.routing_type}")

    return SuryaLayoutResult(blocks=blocks, raw_json=raw_json)


def _parse_surya_json(
    raw_json: Any,
    source_image: np.ndarray,
) -> List[SuryaBlock]:
    """
    Parse le JSON retourné par Surya et extrait les blocs.

    Gère plusieurs formats possibles selon la version du Space :
    - Format A : {"results": [{"bbox": [...], "label": "..."}]}
    - Format B : [{"bbox": [...], "label": "..."}]
    - Format C : {"bboxes": [[...]], "labels": ["..."]}

    Retourne une liste de SuryaBlock triés par position Y puis X.
    """
    import json

    blocks = []
    h, w = source_image.shape[:2]

    if raw_json is None:
        logger.warning("JSON Surya est None")
        return blocks

    # --- Correction du format Gradio filepath ---
    # Gradio peut renvoyer un objet {"__type__": "filepath", "value": "/tmp/xxx.json"}
    # au lieu du JSON brut. On ouvre alors le fichier pour lire le vrai contenu.
    if isinstance(raw_json, dict) and raw_json.get("__type__") == "filepath":
        json_path = raw_json.get("value")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_json = json.load(f)
                logger.info("Fichier JSON Surya lu avec succès depuis le disque.")
        except Exception as e:
            logger.error(f"Impossible de lire le fichier JSON Surya : {e}")
            return []
    # -----------------------------------------------------

    # Normalisation du format en liste de dicts
    items = []
    if isinstance(raw_json, dict):
        if "results" in raw_json:
            items = raw_json["results"]
        elif "bboxes" in raw_json and "labels" in raw_json:
            # Format C : reconstruction
            bboxes = raw_json["bboxes"]
            labels = raw_json["labels"]
            items = [
                {"bbox": b, "label": l}
                for b, l in zip(bboxes, labels)
            ]
        # Nouvelle logique pour gérer le cas où raw_json est un dict avec 'label', 'value', '__type__'
        # mais n'est pas un 'filepath' au sens strict.
        elif "label" in raw_json and "value" in raw_json and "__type__" in raw_json:
            logger.warning(f"Format JSON Surya inattendu mais contient 'label' et 'value' au niveau racine. Tentative de traitement comme un seul bloc. Clés : {list(raw_json.keys())}")
            # Si c'est un seul bloc, on l'ajoute directement à la liste des items
            items.append(raw_json)
        else:
            logger.warning(f"Format JSON Surya inconnu. Clés : {list(raw_json.keys())}")
            return blocks
    elif isinstance(raw_json, list):
        items = raw_json
    else:
        logger.warning(f"Type JSON Surya inattendu : {type(raw_json)}")
        return blocks

    for item in items:
        if not isinstance(item, dict):
            continue

        label = item.get("label", item.get("type", "Unknown"))
        bbox_raw = item.get("bbox", item.get("box", item.get("bounding_box", None)))

        if bbox_raw is None:
            logger.debug(f"Bloc sans bbox ignoré : {item}")
            continue

        # Normalisation de la bbox
        try:
            if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
                xmin, ymin, xmax, ymax = [int(v) for v in bbox_raw]
            elif isinstance(bbox_raw, dict):
                xmin = int(bbox_raw.get("xmin", bbox_raw.get("x0", 0)))
                ymin = int(bbox_raw.get("ymin", bbox_raw.get("y0", 0)))
                xmax = int(bbox_raw.get("xmax", bbox_raw.get("x1", w)))
                ymax = int(bbox_raw.get("ymax", bbox_raw.get("y1", h)))
            else:
                logger.debug(f"Format bbox non reconnu : {bbox_raw}")
                continue
        except (ValueError, TypeError) as e:
            logger.debug(f"Erreur parsing bbox {bbox_raw} : {e}")
            continue

        # Clamp aux dimensions de l'image
        xmin = max(0, min(xmin, w))
        ymin = max(0, min(ymin, h))
        xmax = max(xmin + 1, min(xmax, w))
        ymax = max(ymin + 1, min(ymax, h))

        # Crop du bloc depuis l'image source
        padding = config.SURYA_BLOCK_PADDING
        x1 = max(0, xmin - padding)
        y1 = max(0, ymin - padding)
        x2 = min(w, xmax + padding)
        y2 = min(h, ymax + padding)

        block_img = source_image[y1:y2, x1:x2].copy()

        if block_img.size == 0:
            logger.debug(f"Bloc crop vide ignoré : bbox=({xmin},{ymin},{xmax},{ymax})")
            continue

        blocks.append(SuryaBlock(
            bbox=[xmin, ymin, xmax, ymax],
            label=str(label),
            image=block_img,
        ))

    # Tri spatial : haut → bas, gauche → droite (pour le PDF)
    blocks.sort(key=lambda b: (b.ymin, b.xmin))
    return blocks
