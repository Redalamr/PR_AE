"""
Définition partagée de PipelineResult.

Ce fichier est importé par app.py et layout/ai_pipeline_orchestrator.py
pour éviter une dépendance circulaire.
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from layout.block_detector import Block


@dataclass
class PipelineResult:
    """Résultat complet du pipeline (V2 et V3 partagent cette définition)."""
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
    ocr_avg_confidence: float
