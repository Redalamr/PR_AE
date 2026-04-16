"""Module de détection et classification des blocs."""
# V0/V2 — inchangé
from .block_detector import BlockDetector, Block
from .heuristic_classifier import HeuristicClassifier
from .cnn_classifier import CNNBlockClassifier

# V3 — Pipeline IA
from .api_yolo import detect_whiteboard, crop_whiteboard, DetectionResult
from .api_surya import analyze_layout, SuryaBlock, SuryaLayoutResult
