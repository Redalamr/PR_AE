"""
Configuration centrale du projet.
Tous les chemins, hyperparamètres et constantes sont centralisés ici.
"""

import os
from pathlib import Path

# ============================================
# CHEMINS PROJET
# ============================================
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

for _dir in [MODELS_DIR, DATA_DIR, OUTPUT_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ============================================
# CAPTURE
# ============================================
CAPTURE_KEY = ord('s')
CAPTURE_DIR = PROJECT_ROOT / "captures"
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
WEBCAM_INDEX = 0
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080

# ============================================
# PRÉTRAITEMENT
# ============================================
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)

OTSU_BLUR_KSIZE = 5
ADAPTIVE_BLOCK_SIZE = 15
ADAPTIVE_C = 10

MORPH_KERNEL_SIZE = (3, 3)
MORPH_OPEN_ITER = 1
MORPH_CLOSE_ITER = 1

CANNY_LOW = 50
CANNY_HIGH = 150
CORNER_APPROX_EPSILON = 0.02
MIN_CONTOUR_AREA_RATIO = 0.05

# ============================================
# DÉTECTION DE BLOCS
# ============================================
MIN_BLOCK_AREA = 500
MIN_BLOCK_WIDTH = 30
MIN_BLOCK_HEIGHT = 15
BLOCK_MERGE_DISTANCE_Y = 20
BLOCK_MERGE_DISTANCE_X = 40
BLOCK_PADDING = 5

# ============================================
# CLASSIFIEUR DE BLOCS (ML #1)
# ============================================
CLASSIFIER_INPUT_SIZE = (224, 224)
CLASSIFIER_NUM_CLASSES = 2
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LR = 1e-4
CLASSIFIER_EPOCHS = 20
CLASSIFIER_MODEL_PATH = MODELS_DIR / "block_classifier.pth"

BLOCK_CLASSES = {0: "text", 1: "figure"}
BLOCK_CLASSES_INV = {"text": 0, "figure": 1}

HEURISTIC_ASPECT_RATIO_THRESHOLD = 3.0
HEURISTIC_DARK_DENSITY_THRESHOLD = 0.3

# ============================================
# OCR (ML #2)
# ============================================
DOCTR_MODEL_PATH = MODELS_DIR / "doctr_finetuned_best.pth"
DOCTR_DET_ARCH = "db_resnet50"
DOCTR_RECO_ARCH = "crnn_vgg16_bn"

TROCR_MODEL_NAME = "microsoft/trocr-base-handwritten"

TESSERACT_LANG = "eng"
TESSERACT_CONFIG = "--oem 3 --psm 6"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DOCTR_TRAIN_BATCH_SIZE = 16
DOCTR_TRAIN_LR = 1e-4
DOCTR_TRAIN_EPOCHS = 50
DOCTR_TRAIN_PATIENCE = 5

# ============================================
# AUGMENTATION WHITEBOARD-STYLE
# ============================================
AUG_MOTION_BLUR_LIMIT = (3, 7)
AUG_PERSPECTIVE_SCALE = (0.02, 0.05)
AUG_BRIGHTNESS_LIMIT = (-0.2, 0.2)
AUG_CONTRAST_LIMIT = (-0.2, 0.2)
AUG_GAUSS_NOISE_VAR = (10.0, 50.0)
AUG_DOWNSCALE_MIN = 0.5
AUG_DOWNSCALE_MAX = 0.9

AUG_HEAVY_MOTION_BLUR_LIMIT = (5, 15)
AUG_HEAVY_PERSPECTIVE_SCALE = (0.05, 0.1)
AUG_HEAVY_BRIGHTNESS_LIMIT = (-0.4, 0.3)
AUG_HEAVY_CONTRAST_LIMIT = (-0.3, 0.3)
AUG_HEAVY_GAUSS_NOISE_VAR = (20.0, 80.0)
AUG_HEAVY_DOWNSCALE_MIN = 0.3
AUG_HEAVY_DOWNSCALE_MAX = 0.7

# ============================================
# PDF
# ============================================
PDF_PAGE_WIDTH = 595.27
PDF_PAGE_HEIGHT = 841.89
PDF_MARGIN = 50
PDF_FONT_NAME = "Helvetica"
PDF_FONT_SIZE = 11
PDF_TITLE_FONT_SIZE = 16

# ============================================
# ÉVALUATION
# ============================================
EVAL_CER_TARGET = 0.15
EVAL_WER_TARGET = 0.20
EVAL_F1_TARGET = 0.80
EVAL_PIPELINE_TIME = 60

# ============================================
# MLFLOW
# ============================================
MLFLOW_TRACKING_URI = str(LOGS_DIR / "mlruns")
MLFLOW_EXPERIMENT_CLASSIFIER = "block_classifier"
MLFLOW_EXPERIMENT_OCR = "ocr_finetuning"

# ============================================
# V2 — CORRECTION LLM
# ============================================
LLM_DEFAULT_PROVIDER = "simulate"          # "openai", "anthropic", "simulate"
LLM_DEFAULT_MODEL_OPENAI = "gpt-4o-mini"
LLM_DEFAULT_MODEL_ANTHROPIC = "claude-3-5-sonnet-20241022"
LLM_TEMPERATURE = 0.0                      # Déterministe pour la correction

# ============================================
# V2 — LATEX-OCR
# ============================================
LATEX_OCR_BACKEND = "pix2tex"              # "pix2tex", "nougat", "simulate"
NOUGAT_MODEL_NAME = "facebook/nougat-base"

# Labels de blocs reconnus comme "mathématiques"
MATH_BLOCK_LABELS = {"equation", "math", "formula", "latex", "mathematical"}

# ============================================
# V2 — CLASSIFICATION ÉTENDUE
# ============================================
# Extension du dictionnaire V0 pour supporter les blocs maths
BLOCK_CLASSES_V2 = {0: "text", 1: "figure", 2: "equation"}
BLOCK_CLASSES_V2_INV = {"text": 0, "figure": 1, "equation": 2}

# ============================================
# V2 — STREAMLIT
# ============================================
STREAMLIT_MAX_UPLOAD_MB = 20
STREAMLIT_DEFAULT_BINARIZATION = "adaptive_clahe"
