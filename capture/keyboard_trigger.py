"""
Étape 1 — Capture clavier
Appui sur 'S' → capture la frame courante de la webcam et la sauvegarde.
Supporte aussi le chargement d'images statiques pour les tests.

Stack : OpenCV VideoCapture + cv2.imwrite()
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import logging

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


class KeyboardCapture:
    """
    Gestionnaire de capture d'images via webcam ou fichier statique.

    Usage webcam:
        cap = KeyboardCapture()
        cap.start_live_capture()

    Usage statique:
        cap = KeyboardCapture()
        image = cap.load_static_image("path/to/image.jpg")
    """

    def __init__(
        self,
        webcam_index: int = config.WEBCAM_INDEX,
        save_dir: Optional[Path] = None,
        resolution: Tuple[int, int] = (config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT),
    ):
        self.webcam_index = webcam_index
        self.save_dir = save_dir or config.CAPTURE_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.resolution = resolution
        self._cap: Optional[cv2.VideoCapture] = None
        logger.info(f"KeyboardCapture initialisé — save_dir={self.save_dir}")

    def _open_webcam(self) -> cv2.VideoCapture:
        """Ouvre la webcam et configure la résolution."""
        cap = cv2.VideoCapture(self.webcam_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Impossible d'ouvrir la webcam (index={self.webcam_index}). "
                "Vérifiez qu'elle est connectée."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        logger.info(
            f"Webcam ouverte : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        )
        return cap

    def start_live_capture(self) -> Optional[np.ndarray]:
        """
        Ouvre la webcam en mode live.
        - 'S' → capture et sauvegarde la frame courante.
        - 'Q' → ferme la webcam et retourne la dernière capture.
        """
        self._cap = self._open_webcam()
        last_capture = None
        logger.info("Mode live — Appuyez 'S' pour capturer, 'Q' pour quitter")

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning("Échec de lecture webcam")
                    continue

                display = frame.copy()
                cv2.putText(
                    display, "[S] Capturer  |  [Q] Quitter",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                cv2.imshow("Capture Tableau Blanc — V0", display)

                key = cv2.waitKey(1) & 0xFF
                if key == config.CAPTURE_KEY:
                    saved_path = self._save_frame(frame)
                    last_capture = frame.copy()
                    logger.info(f"Image capturée → {saved_path}")
                    flash = np.ones_like(frame, dtype=np.uint8) * 255
                    cv2.imshow("Capture Tableau Blanc — V0", flash)
                    cv2.waitKey(100)
                elif key == ord('q'):
                    break
        finally:
            self._release()

        return last_capture

    def capture_single_frame(self) -> np.ndarray:
        """Capture une seule frame depuis la webcam."""
        self._cap = self._open_webcam()
        try:
            for _ in range(5):
                self._cap.read()
            ret, frame = self._cap.read()
            if not ret:
                raise RuntimeError("Échec de capture d'une frame")
            saved_path = self._save_frame(frame)
            logger.info(f"Frame capturée → {saved_path}")
            return frame
        finally:
            self._release()

    def load_static_image(self, image_path: str) -> np.ndarray:
        """Charge une image depuis un fichier."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image non trouvée : {image_path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Impossible de lire l'image : {image_path}")
        logger.info(f"Image statique chargée : {path.name} ({image.shape})")
        return image

    def _save_frame(self, frame: np.ndarray) -> Path:
        """Sauvegarde une frame avec un nom horodaté."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"capture_{timestamp}.png"
        save_path = self.save_dir / filename
        cv2.imwrite(str(save_path), frame)
        return save_path

    def _release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        cv2.destroyAllWindows()

    def __del__(self):
        self._release()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser(description="Capture d'image — V0")
    parser.add_argument("--static", type=str, default=None,
                        help="Chemin vers une image statique (sinon webcam live)")
    args = parser.parse_args()
    cap = KeyboardCapture()
    if args.static:
        img = cap.load_static_image(args.static)
        print(f"Image chargée : {img.shape}")
    else:
        img = cap.start_live_capture()
        if img is not None:
            print(f"Dernière capture : {img.shape}")
