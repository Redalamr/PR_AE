"""Module de détection et classification des blocs."""
# V0/V2 — inchangé
from .block_detector import BlockDetector, Block
from .heuristic_classifier import HeuristicClassifier
from .cnn_classifier import CNNBlockClassifier

# V3 — Pipeline IA (import conditionnel pour éviter crash si gradio non installé)
try:
    from .api_yolo import detect_whiteboard, crop_whiteboard, DetectionResult
    from .api_surya import analyze_layout, SuryaBlock, SuryaLayoutResult
    HAS_AI_PIPELINE = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(
        f"Pipeline IA non disponible ({e}). "
        "Installez gradio_client: pip install gradio_client>=0.16.0"
    )
    HAS_AI_PIPELINE = False
