"""
Étape 6 — Génération PDF.

Assemble le texte reconnu + les figures recadrées dans un PDF structuré
et indexable, en préservant l'ordre spatial du tableau.

Stack : ReportLab
"""

import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import logging
import tempfile

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
)

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


class PDFGenerator:
    """
    Génère un PDF structuré à partir du texte reconnu et des figures.

    Usage :
        gen = PDFGenerator()
        gen.add_text("Texte reconnu par OCR...")
        gen.add_figure(figure_image)
        gen.save("output.pdf")
    """

    def __init__(
        self, title: str = "Tableau Blanc — Capture OCR",
        author: str = "Pipeline V0 — ESAIP",
        output_dir: Optional[Path] = None,
    ):
        self.title = title
        self.author = author
        self.output_dir = output_dir or config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._elements = []
        self._temp_files = []
        self.styles = getSampleStyleSheet()
        self.styles["Normal"].fontSize = config.PDF_FONT_SIZE
        self.styles["Normal"].leading = config.PDF_FONT_SIZE * 1.4
        logger.info(f"PDFGenerator — output_dir={self.output_dir}")

    def add_title(self, text: str):
        """Ajoute un titre au PDF."""
        self._elements.append(Paragraph(text, self.styles["Heading1"]))
        self._elements.append(Spacer(1, 12))

    def add_subtitle(self, text: str):
        """Ajoute un sous-titre."""
        self._elements.append(Paragraph(text, self.styles["Heading2"]))
        self._elements.append(Spacer(1, 8))

    def add_text(self, text: str):
        """Ajoute un bloc de texte reconnu."""
        if not text.strip():
            return
        safe_text = (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        for line in safe_text.split("\n"):
            if line.strip():
                self._elements.append(Paragraph(line.strip(), self.styles["Normal"]))
        self._elements.append(Spacer(1, 10))

    def add_figure(
        self, image: np.ndarray, caption: str = "",
        max_width: float = 400, max_height: float = 300,
    ):
        """Ajoute une figure (image) au PDF."""
        tmp = tempfile.NamedTemporaryFile(
            suffix=".png", delete=False, dir=str(self.output_dir)
        )
        cv2.imwrite(tmp.name, image)
        self._temp_files.append(tmp.name)

        h, w = image.shape[:2]
        ratio = min(max_width / w, max_height / h)
        self._elements.append(RLImage(tmp.name, width=w * ratio, height=h * ratio))

        if caption:
            self._elements.append(Paragraph(caption, self.styles["Italic"]))
        self._elements.append(Spacer(1, 12))

    def add_block(self, block, ocr_result=None):
        """Ajoute un bloc (texte ou figure) au PDF."""
        if block.label == "text" and ocr_result:
            self.add_text(ocr_result.text)
        elif block.label == "figure" and block.image is not None:
            self.add_figure(block.image, caption="[Figure détectée]")

    def save(self, filename: Optional[str] = None) -> Path:
        """Génère et sauvegarde le PDF."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tableau_blanc_{timestamp}.pdf"

        # Forcer le nom de fichier uniquement, sans chemin parent
        safe_name = Path(filename).name
        output_path = self.output_dir / safe_name

        header = [
            Paragraph(self.title, self.styles["Title"]),
            Paragraph(
                f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}",
                self.styles["Normal"],
            ),
            Spacer(1, 20),
        ]

        doc = SimpleDocTemplate(
            str(output_path), pagesize=A4,
            leftMargin=config.PDF_MARGIN, rightMargin=config.PDF_MARGIN,
            topMargin=config.PDF_MARGIN, bottomMargin=config.PDF_MARGIN,
            title=self.title, author=self.author,
        )
        doc.build(header + self._elements)
        logger.info(f"PDF généré → {output_path}")
        self._cleanup()
        return output_path

    def _cleanup(self):
        for tmp_file in self._temp_files:
            try:
                os.unlink(tmp_file)
            except OSError:
                pass
        self._temp_files.clear()

    def __del__(self):
        self._cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = PDFGenerator(title="Test PDF Generator")
    gen.add_title("Capture du Tableau Blanc")
    gen.add_text("Ceci est un texte reconnu par le pipeline OCR.")

    fig = np.ones((200, 300, 3), dtype=np.uint8) * 255
    cv2.circle(fig, (150, 100), 50, (0, 0, 0), 2)
    gen.add_figure(fig, caption="Cercle détecté")

    path = gen.save("test_output.pdf")
    print(f"PDF de test : {path}")
