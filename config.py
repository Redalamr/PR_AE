"""
Configuration centrale du projet.
Tous les chemins, hyperparamètres et constantes sont centralisés ici.
"""

import os
from pathlib import Path

# Chargement basique des variables d'environnement (si .env présent localement)
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            if _line.strip() and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.strip().split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

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
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_SIZE = (8, 8)

OTSU_BLUR_KSIZE = 7
ADAPTIVE_BLOCK_SIZE = 35
ADAPTIVE_C = 10

MORPH_KERNEL_SIZE = (3, 3)
MORPH_OPEN_ITER = 1
MORPH_CLOSE_ITER = 1
SUBTRACT_BACKGROUND = True

CANNY_LOW = 30
CANNY_HIGH = 90
CORNER_APPROX_EPSILON = 0.03
MIN_CONTOUR_AREA_RATIO = 0.02

# ============================================
# DÉTECTION DE BLOCS
# ============================================
MIN_BLOCK_AREA = 300
MIN_BLOCK_WIDTH = 15
MIN_BLOCK_HEIGHT = 10
BLOCK_MERGE_DISTANCE_Y = 30
BLOCK_MERGE_DISTANCE_X = 60
BLOCK_PADDING = 6

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

HEURISTIC_ASPECT_RATIO_THRESHOLD = 2.5
HEURISTIC_DARK_DENSITY_THRESHOLD = 0.05

# ============================================
# OCR (ML #2)
# ============================================
DOCTR_MODEL_PATH = MODELS_DIR / "doctr_finetuned_best.pth"
DOCTR_DET_ARCH = "db_resnet50"
DOCTR_RECO_ARCH = "crnn_vgg16_bn"

TROCR_MODEL_NAME = "microsoft/trocr-base-handwritten"

TESSERACT_LANG = "eng"
TESSERACT_CONFIG = "--oem 3 --psm 11"
import sys as _sys

# ── Chemin Tesseract selon l'OS ──
if _sys.platform == "win32":
    TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
elif _sys.platform == "darwin":
    # Homebrew : brew install tesseract
    TESSERACT_CMD = "/usr/local/bin/tesseract"
else:
    # Linux : sudo apt-get install tesseract-ocr
    TESSERACT_CMD = "tesseract"

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

# ============================================
# V3 — PIPELINE IA (YOLO-World + Surya)
# ============================================
HF_API_KEY_DEFAULT = os.getenv("HF_API_KEY", "")

# ID du modèle pour la détection de tableaux (Inference API)
# facebook/detr-resnet-50 est un modèle standard robuste supporté par l'API
YOLO_WORLD_MODEL_ID = "facebook/detr-resnet-50"

# Endpoint HuggingFace Inference API
HF_INFERENCE_API_URL = f"https://api-inference.huggingface.co/models/{YOLO_WORLD_MODEL_ID}"

# Labels de détection (DETR utilise les classes COCO, 'dining table' est souvent le plus proche pour un tableau blanc)
YOLO_WHITEBOARD_LABELS = ["dining table", "laptop", "tv"]

# Seuil de confiance minimum pour accepter une détection YOLO
YOLO_CONFIDENCE_THRESHOLD = 0.20

# Space Gradio Surya (public, pas de token requis)
SURYA_SPACE_ID = "xiaoyao9184/surya"
SURYA_API_NAME = "/layout_det_img"

# Timeout (secondes) pour les appels API distants
API_CALL_TIMEOUT = 60

# Labels Surya reconnus comme "texte" → routage vers OCR
SURYA_TEXT_LABELS = {"Text", "TextInlineMath", "Caption", "Footnote", "SectionHeader", "Title", "ListItem"}

# Labels Surya reconnus comme "mathématiques" → routage vers LaTeX-OCR
SURYA_MATH_LABELS = {"Formula", "Equation", "Math"}

# Labels Surya reconnus comme "figure" → sauvegarde image
SURYA_FIGURE_LABELS = {"Figure", "Image", "Picture"}

# Padding (pixels) appliqué lors du crop de chaque bloc Surya
SURYA_BLOCK_PADDING = 8

# ============================================
# V4 — PIPELINE TOUT-EN-UN (Gemini 2.5)
# ============================================
GEMINI_API_KEY_DEFAULT = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_ID = "gemini-2.5-flash"
GEMINI_SYSTEM_PROMPT = """Tu es un expert en OCR avec une spécialisation en mathématiques et structuration de document. 
Ta tâche est d'analyser l'image de tableau blanc fournie.
Extrais tout le texte et les formules mathématiques visibles. 
Mets les formules mathématiques en format LaTeX entourées de symboles '$' ou '$$'.
Structure la sortie en syntaxe Markdown lisible, avec des titres, des listes si pertinents, et transcris fidèlement la logique du document.
Si des figures de dessins, graphes ou schémas sont présents, ajoute une courte balise descriptive du genre : [FIGURE: description pertinente].
Ne rajoute pas de textes superflus, ta réponse entière doit servir de transcription finale."""
