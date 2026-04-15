"""
Pipeline V2 — Application Streamlit.

Interface web complète pour le pipeline OCR de tableau blanc.
Encapsule les modules V0 existants + nouvelles fonctionnalités V2 :
  - Upload & visualisation d'images
  - Pipeline OCR complet (preprocessing → détection → classification → OCR)
  - Correction post-OCR via LLM (OpenAI / Anthropic / simulation)
  - Reconnaissance LaTeX pour les blocs mathématiques
  - Édition manuelle du texte extrait
  - Export PDF propre

Usage :
    streamlit run app.py
    streamlit run app.py -- --debug
"""

import streamlit as st
import cv2
import numpy as np
import time
import logging
import tempfile
import io
from pathlib import Path
from PIL import Image
from datetime import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass

# ────────────────────────────────────────────
# Imports du pipeline V0 existant
# ────────────────────────────────────────────
import config
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.enhance import BinarizationMethod
from layout.block_detector import BlockDetector, Block
from layout.heuristic_classifier import HeuristicClassifier

# ────────────────────────────────────────────
# Imports V2 — Nouveaux modules
# ────────────────────────────────────────────
from llm.corrector import LLMCorrector
from ocr.latex_ocr_engine import LatexOCREngine, is_math_block

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration Streamlit
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="OCR Tableau Blanc — V2",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS custom — UI premium
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    /* ── Variables ── */
    :root {
        --accent: #6C63FF;
        --accent-light: #a29bfe;
        --bg-dark: #0E1117;
        --bg-card: #1a1d24;
        --border: #2d3139;
        --text-primary: #FAFAFA;
        --text-secondary: #8b949e;
        --success: #2dce89;
        --warning: #fb6340;
        --info: #11cdef;
    }

    /* ── Global ── */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #161b22 50%, #0E1117 100%);
    }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.15) 0%, rgba(162, 155, 254, 0.08) 100%);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    .main-header h1 {
        background: linear-gradient(135deg, #6C63FF, #a29bfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: var(--text-secondary);
        font-size: 1rem;
        margin: 0;
    }

    /* ── Metric cards ── */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        flex: 1;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--accent-light);
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Status badges ── */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-success { background: rgba(45, 206, 137, 0.15); color: var(--success); border: 1px solid rgba(45, 206, 137, 0.3); }
    .badge-info { background: rgba(17, 205, 239, 0.15); color: var(--info); border: 1px solid rgba(17, 205, 239, 0.3); }
    .badge-warning { background: rgba(251, 99, 64, 0.15); color: var(--warning); border: 1px solid rgba(251, 99, 64, 0.3); }

    /* ── Section headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }
    .section-header h3 {
        margin: 0;
        font-weight: 600;
    }

    /* ── Block card ── */
    .block-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s;
    }
    .block-card:hover {
        border-color: var(--accent);
    }
    .block-type-text { border-left: 3px solid var(--success); }
    .block-type-equation { border-left: 3px solid var(--info); }
    .block-type-figure { border-left: 3px solid var(--warning); }

    /* ── Divider ── */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 2rem 0;
    }

    /* ── Hide Streamlit defaults ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Custom scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fonctions utilitaires
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource
def load_ocr_engine(engine_type: str):
    """Charge et cache le moteur OCR sélectionné."""
    if engine_type == "doctr":
        from ocr.doctr_engine import DocTREngine
        return DocTREngine()
    elif engine_type == "trocr":
        from ocr.trocr_engine import TrOCREngine
        return TrOCREngine()
    else:
        from ocr.tesseract_engine import TesseractEngine
        return TesseractEngine()


@st.cache_resource
def load_latex_engine(backend: str):
    """Charge et cache le moteur LaTeX-OCR."""
    return LatexOCREngine(backend=backend)


@st.cache_resource
def load_llm_corrector(provider: str, api_key: Optional[str] = None):
    """Charge et cache le correcteur LLM."""
    return LLMCorrector(provider=provider, api_key=api_key if api_key else None)


def uploaded_file_to_cv2(uploaded_file) -> np.ndarray:
    """Convertit un fichier uploadé Streamlit → image OpenCV (BGR)."""
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    uploaded_file.seek(0)  # Reset pour relecture éventuelle
    return image


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convertit une image OpenCV (BGR) → PIL (RGB)."""
    if len(image.shape) == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def generate_pdf_bytes(
    text: str,
    title: str = "Capture du Tableau Blanc",
    figure_images: Optional[List[np.ndarray]] = None,
    latex_blocks: Optional[List[str]] = None,
) -> bytes:
    """
    Génère un PDF en mémoire et renvoie les bytes.

    Utilise le PDFGenerator du Pipeline V0 existant (output/pdf_generator.py)
    mais capture les bytes au lieu de sauvegarder sur disque.
    """
    from output.pdf_generator import PDFGenerator

    generator = PDFGenerator(title=title)
    generator.add_title(title)

    timestamp = datetime.now().strftime("%d/%m/%Y à %H:%M:%S")
    generator.add_text(f"Généré le {timestamp}")

    # Ajout du texte principal
    if text.strip():
        generator.add_subtitle("Texte Reconnu")
        generator.add_text(text)

    # Ajout des formules LaTeX (comme texte brut dans le PDF)
    if latex_blocks:
        generator.add_subtitle("Formules Mathématiques")
        for i, latex in enumerate(latex_blocks, 1):
            generator.add_text(f"Formule {i} : ${latex}$")

    # Ajout des figures
    if figure_images:
        generator.add_subtitle("Figures Détectées")
        for i, fig in enumerate(figure_images, 1):
            generator.add_figure(fig, caption=f"Figure {i}")

    # Sauvegarde temporaire → lecture bytes
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    generator.output_dir = tmp_path.parent
    pdf_path = generator.save(tmp_path.name)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Nettoyage
    try:
        pdf_path.unlink()
    except OSError:
        pass

    return pdf_bytes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pipeline de traitement principal
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class PipelineResult:
    """Résultat complet du pipeline V2."""
    raw_text: str
    corrected_text: str
    latex_formulas: List[str]
    figure_images: List[np.ndarray]
    blocks: List[Block]
    processing_time: float
    block_count: int
    text_block_count: int
    math_block_count: int
    figure_block_count: int
    corrections_count: int
    binary_image: np.ndarray


def run_pipeline(
    image: np.ndarray,
    ocr_engine_type: str,
    latex_backend: str,
    llm_provider: str,
    llm_api_key: Optional[str],
    binarization: str,
    skip_perspective: bool,
    use_cnn: bool,
    enable_llm: bool,
    enable_latex: bool,
) -> PipelineResult:
    """
    Exécute le pipeline V2 complet sur une image.

    Flux :
        Image → Preprocessing → Block Detection → Classification
        → OCR (texte) / LaTeX-OCR (maths) → Correction LLM → Résultat
    """
    start_time = time.time()

    # ── 1. Prétraitement ──
    preprocessing = PreprocessingPipeline(
        binarization_method=BinarizationMethod(binarization),
        skip_perspective=skip_perspective,
    )
    binary = preprocessing.run(image)

    # ── 2. Détection de blocs ──
    detector = BlockDetector()
    blocks = detector.detect(binary)

    # ── 3. Classification ──
    if use_cnn:
        try:
            from layout.cnn_classifier import CNNBlockClassifier
            classifier = CNNBlockClassifier()
        except Exception:
            classifier = HeuristicClassifier()
    else:
        classifier = HeuristicClassifier()

    for block in blocks:
        if block.image is not None:
            label, conf = classifier.classify(block.image)
            block.label = label
            block.confidence = conf

    # ────────────────────────────────────────
    # POINT DE BRANCHEMENT V2 :
    # Ici on route chaque bloc vers le bon moteur OCR
    # selon son label de classification.
    #
    # Pour ajouter un nouveau type de bloc (ex: "code", "diagram"),
    # il suffit d'ajouter un elif et le moteur correspondant.
    # ────────────────────────────────────────

    # ── 4. OCR / LaTeX-OCR (routage par type) ──
    ocr_engine = load_ocr_engine(ocr_engine_type)
    latex_engine = load_latex_engine(latex_backend) if enable_latex else None

    text_parts = []
    latex_formulas = []
    figure_images = []

    for block in blocks:
        if block.image is None:
            continue

        # Route : Bloc mathématique → LaTeX-OCR
        if enable_latex and is_math_block(block.label):
            result = latex_engine.recognize(block.image)
            latex_formulas.append(result.text)

        # Route : Bloc texte → OCR classique
        elif block.label == "text":
            result = ocr_engine.recognize(block.image)
            text_parts.append(result.text)

        # Route : Bloc figure → on garde l'image
        elif block.label == "figure":
            figure_images.append(block.image)

    raw_text = "\n\n".join(text_parts)

    # ── 5. Correction LLM ──
    corrections_count = 0
    if enable_llm and raw_text.strip():
        corrector = load_llm_corrector(llm_provider, llm_api_key)
        correction_result = corrector.correct(raw_text)
        corrected_text = correction_result.corrected_text
        corrections_count = correction_result.corrections_count
    else:
        corrected_text = raw_text

    # ── Métriques ──
    processing_time = time.time() - start_time
    text_count = sum(1 for b in blocks if b.label == "text")
    math_count = sum(1 for b in blocks if is_math_block(b.label))
    fig_count = sum(1 for b in blocks if b.label == "figure")

    return PipelineResult(
        raw_text=raw_text,
        corrected_text=corrected_text,
        latex_formulas=latex_formulas,
        figure_images=figure_images,
        blocks=blocks,
        processing_time=processing_time,
        block_count=len(blocks),
        text_block_count=text_count,
        math_block_count=math_count,
        figure_block_count=fig_count,
        corrections_count=corrections_count,
        binary_image=binary,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UI — Sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # ── Moteur OCR ──
    st.markdown("### 🔤 Moteur OCR")
    ocr_engine_type = st.selectbox(
        "Moteur principal",
        options=["tesseract", "doctr", "trocr"],
        index=0,
        help="Tesseract = rapide, docTR = précis (fine-tuné), TrOCR = DL pré-entraîné",
    )

    # ── Classifieur ──
    st.markdown("### 🏷️ Classifieur")
    use_cnn = st.checkbox("Utiliser le CNN (sinon heuristique)", value=False)

    # ── Prétraitement ──
    st.markdown("### 🖼️ Prétraitement")
    binarization = st.selectbox(
        "Binarisation",
        options=["adaptive_clahe", "adaptive", "otsu_clahe", "otsu"],
        index=0,
    )
    skip_perspective = st.checkbox("Ignorer la correction de perspective", value=True)

    st.markdown("---")

    # ── LaTeX-OCR (V2) ──
    st.markdown("### 📐 LaTeX-OCR")
    enable_latex = st.checkbox("Activer LaTeX-OCR (formules)", value=True)
    latex_backend = st.selectbox(
        "Backend",
        options=["simulate", "pix2tex", "nougat"],
        index=0,
        help="simulate = test sans modèle, pix2tex = formules isolées, nougat = pages complètes",
    )

    st.markdown("---")

    # ── Correction LLM (V2) ──
    st.markdown("### 🤖 Correction LLM")
    enable_llm = st.checkbox("Activer la correction post-OCR", value=True)
    llm_provider = st.selectbox(
        "Provider",
        options=["simulate", "openai", "anthropic"],
        index=0,
        help="simulate = regex offline, openai/anthropic = API cloud",
    )

    llm_api_key = None
    if llm_provider in ("openai", "anthropic"):
        llm_api_key = st.text_input(
            f"🔑 Clé API {llm_provider.title()}",
            type="password",
            help="Alternativement, exportez OPENAI_API_KEY ou ANTHROPIC_API_KEY",
        )

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:#8b949e; font-size:0.75rem;">'
        "Pipeline OCR Tableau Blanc · V2<br>ESAIP · IR4-S8 · 2025–2026"
        "</p>",
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UI — Main Content
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Header ──
st.markdown("""
<div class="main-header">
    <h1>🧠 OCR Tableau Blanc — V2</h1>
    <p>Pipeline intelligent : Prétraitement → Détection → OCR / LaTeX-OCR → Correction LLM → Export PDF</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ──
col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "📤 Uploadez une image de tableau blanc",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Formats acceptés : JPG, PNG, BMP, TIFF",
    )

with col_info:
    if uploaded_file:
        file_size = len(uploaded_file.getvalue()) / 1024  # KB
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 0.5rem;">
            <div class="label">Fichier chargé</div>
            <div class="value" style="font-size: 1rem;">{uploaded_file.name}</div>
            <div class="label">{file_size:.0f} KB</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card" style="margin-top: 0.5rem;">
            <div class="value" style="font-size: 1.2rem;">📋</div>
            <div class="label">En attente d'une image</div>
        </div>
        """, unsafe_allow_html=True)


# ── Traitement ──
if uploaded_file is not None:
    image = uploaded_file_to_cv2(uploaded_file)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Visualisation de l'image originale
    st.markdown("""
    <div class="section-header">
        <h3>🖼️ Image Originale</h3>
    </div>
    """, unsafe_allow_html=True)

    col_orig, col_preview = st.columns(2)
    with col_orig:
        st.image(cv2_to_pil(image), caption="Image uploadée", use_container_width=True)

    with col_preview:
        # Aperçu rapide des dimensions
        h, w = image.shape[:2]
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="value">{w}×{h}</div>
                <div class="label">Résolution</div>
            </div>
            <div class="metric-card">
                <div class="value">{image.shape[2] if len(image.shape) == 3 else 1}</div>
                <div class="label">Canaux</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Badges de configuration active
        badges = []
        badges.append(f'<span class="badge badge-info">{ocr_engine_type.upper()}</span>')
        if enable_latex:
            badges.append(f'<span class="badge badge-success">LaTeX: {latex_backend}</span>')
        if enable_llm:
            badges.append(f'<span class="badge badge-warning">LLM: {llm_provider}</span>')
        st.markdown(f'<div style="margin-top: 1rem;">{"  ".join(badges)}</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Bouton de lancement ──
    col_btn, col_spacer = st.columns([1, 2])
    with col_btn:
        process_btn = st.button(
            "🚀 Lancer le traitement OCR",
            use_container_width=True,
            type="primary",
        )

    if process_btn:
        # ── Barre de progression ──
        progress_bar = st.progress(0, text="Initialisation du pipeline…")

        with st.spinner("⏳ Traitement en cours…"):
            progress_bar.progress(10, text="Prétraitement de l'image…")

            result = run_pipeline(
                image=image,
                ocr_engine_type=ocr_engine_type,
                latex_backend=latex_backend,
                llm_provider=llm_provider,
                llm_api_key=llm_api_key,
                binarization=binarization,
                skip_perspective=skip_perspective,
                use_cnn=use_cnn,
                enable_llm=enable_llm,
                enable_latex=enable_latex,
            )

            progress_bar.progress(100, text="✅ Traitement terminé !")

        # Stocker le résultat en session
        st.session_state["pipeline_result"] = result
        time.sleep(0.5)
        progress_bar.empty()

    # ── Affichage des résultats ──
    if "pipeline_result" in st.session_state:
        result = st.session_state["pipeline_result"]

        # ── Métriques ──
        st.markdown("""
        <div class="section-header">
            <h3>📊 Résultats du Pipeline</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="value">{result.processing_time:.1f}s</div>
                <div class="label">Temps total</div>
            </div>
            <div class="metric-card">
                <div class="value">{result.block_count}</div>
                <div class="label">Blocs détectés</div>
            </div>
            <div class="metric-card">
                <div class="value">{result.text_block_count}</div>
                <div class="label">Blocs texte</div>
            </div>
            <div class="metric-card">
                <div class="value">{result.math_block_count}</div>
                <div class="label">Blocs maths</div>
            </div>
            <div class="metric-card">
                <div class="value">{result.figure_block_count}</div>
                <div class="label">Figures</div>
            </div>
            <div class="metric-card">
                <div class="value">{result.corrections_count}</div>
                <div class="label">Corrections LLM</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Image binarisée + blocs ──
        col_bin, col_blocks = st.columns(2)

        with col_bin:
            st.markdown("#### 🔲 Image Binarisée")
            st.image(result.binary_image, caption="Sortie du prétraitement", use_container_width=True)

        with col_blocks:
            st.markdown("#### 🧩 Blocs Détectés")
            # Visualisation des blocs
            detector = BlockDetector()
            vis = detector.visualize(image, result.blocks)
            st.image(cv2_to_pil(vis), caption=f"{result.block_count} blocs", use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Texte extrait (éditable) ──
        st.markdown("""
        <div class="section-header">
            <h3>📝 Texte Extrait</h3>
        </div>
        """, unsafe_allow_html=True)

        # Onglets pour texte brut vs corrigé
        tab_corrected, tab_raw, tab_latex = st.tabs(["✅ Texte Corrigé", "📄 Texte Brut OCR", "📐 Formules LaTeX"])

        with tab_corrected:
            st.markdown(
                '<span class="badge badge-success">Après correction LLM</span>'
                if enable_llm else
                '<span class="badge badge-info">Sans correction LLM</span>',
                unsafe_allow_html=True,
            )
            edited_text = st.text_area(
                "Éditez le texte si nécessaire :",
                value=result.corrected_text,
                height=350,
                key="edited_text",
                help="Ce texte sera utilisé pour la génération du PDF",
            )

        with tab_raw:
            st.markdown('<span class="badge badge-warning">Texte brut OCR — non corrigé</span>', unsafe_allow_html=True)
            st.text_area(
                "Texte brut :",
                value=result.raw_text,
                height=350,
                disabled=True,
                key="raw_text_display",
            )

        with tab_latex:
            if result.latex_formulas:
                for i, formula in enumerate(result.latex_formulas, 1):
                    st.markdown(f"""
                    <div class="block-card block-type-equation">
                        <strong>Formule {i}</strong><br>
                        <code>{formula}</code>
                    </div>
                    """, unsafe_allow_html=True)
                    # Rendu LaTeX si possible
                    try:
                        st.latex(formula)
                    except Exception:
                        pass
            else:
                st.info("Aucune formule mathématique détectée dans cette image.")

        # ── Figures ──
        if result.figure_images:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="section-header">
                <h3>🖼️ Figures Détectées</h3>
            </div>
            """, unsafe_allow_html=True)

            fig_cols = st.columns(min(len(result.figure_images), 3))
            for i, fig_img in enumerate(result.figure_images):
                with fig_cols[i % 3]:
                    st.image(
                        cv2_to_pil(fig_img),
                        caption=f"Figure {i+1}",
                        use_container_width=True,
                    )

        # ── Export PDF ──
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
            <h3>📥 Export PDF</h3>
        </div>
        """, unsafe_allow_html=True)

        col_pdf_btn, col_pdf_download = st.columns(2)

        with col_pdf_btn:
            pdf_title = st.text_input(
                "Titre du PDF",
                value="Capture du Tableau Blanc",
                key="pdf_title",
            )

        with col_pdf_download:
            # Calcul du texte final (depuis la zone d'édition)
            final_text = st.session_state.get("edited_text", result.corrected_text)

            try:
                pdf_bytes = generate_pdf_bytes(
                    text=final_text,
                    title=pdf_title,
                    figure_images=result.figure_images if result.figure_images else None,
                    latex_blocks=result.latex_formulas if result.latex_formulas else None,
                )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="📥 Télécharger le PDF",
                    data=pdf_bytes,
                    file_name=f"tableau_blanc_{timestamp}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                )
            except Exception as e:
                st.error(f"Erreur lors de la génération du PDF : {e}")


else:
    # ── État initial — pas d'image ──
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card" style="min-height: 160px;">
            <div class="value" style="font-size: 2.5rem;">📤</div>
            <div class="label" style="margin-top: 0.5rem;">Étape 1</div>
            <div style="color: #FAFAFA; font-size: 0.9rem; margin-top: 0.5rem;">Uploadez une image</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style="min-height: 160px;">
            <div class="value" style="font-size: 2.5rem;">🚀</div>
            <div class="label" style="margin-top: 0.5rem;">Étape 2</div>
            <div style="color: #FAFAFA; font-size: 0.9rem; margin-top: 0.5rem;">Lancez le traitement</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card" style="min-height: 160px;">
            <div class="value" style="font-size: 2.5rem;">📥</div>
            <div class="label" style="margin-top: 0.5rem;">Étape 3</div>
            <div style="color: #FAFAFA; font-size: 0.9rem; margin-top: 0.5rem;">Téléchargez le PDF</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Fonctionnalités V2 ──
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### ✨ Fonctionnalités V2")

    feat_cols = st.columns(3)
    with feat_cols[0]:
        st.markdown("""
        **🤖 Correction LLM**
        > Le texte brut OCR est automatiquement corrigé par un LLM
        > spécialisé en vocabulaire IA/ML. Supporte OpenAI, Anthropic,
        > et un mode simulation offline.
        """)

    with feat_cols[1]:
        st.markdown("""
        **📐 LaTeX-OCR**
        > Les formules mathématiques sont détectées et converties
        > en LaTeX via pix2tex ou Nougat (Meta). Rendu interactif
        > directement dans l'interface.
        """)

    with feat_cols[2]:
        st.markdown("""
        **📥 Export PDF**
        > Le texte corrigé, les formules et les figures sont
        > assemblés dans un PDF structuré et téléchargeable.
        > Édition manuelle avant export.
        """)
