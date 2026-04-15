"""
Pipeline d'augmentation whiteboard-style.

Transformations sur images IAM pour simuler les conditions tableau blanc :
- Flou de mouvement (MotionBlur)
- Distorsion de perspective légère (Perspective)
- Variations d'éclairage (RandomBrightnessContrast, RandomShadow)
- Bruit gaussien (GaussNoise)
- Dégradation de résolution (Downscale)

Stack : albumentations
"""

import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from typing import Optional
import json
import logging

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


class WhiteboardAugmentor:
    """
    Pipeline d'augmentation synthétique — domain adaptation whiteboard.

    Deux modes :
    - 'light' : augmentation légère (Run C)
    - 'heavy' : augmentation agressive (Run D & E)
    """

    def __init__(self, mode: str = "light"):
        assert mode in ("light", "heavy"), f"Mode invalide : {mode}"
        self.mode = mode
        self.transform = self._build_pipeline(mode)
        logger.info(f"WhiteboardAugmentor — mode={mode}")

    def _build_pipeline(self, mode: str) -> A.Compose:
        """Construit le pipeline albumentations selon le mode."""
        if mode == "light":
            return A.Compose([
                A.MotionBlur(blur_limit=config.AUG_MOTION_BLUR_LIMIT, p=0.3),
                A.Perspective(scale=config.AUG_PERSPECTIVE_SCALE, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=config.AUG_BRIGHTNESS_LIMIT,
                    contrast_limit=config.AUG_CONTRAST_LIMIT, p=0.5,
                ),
                A.GaussNoise(var_limit=config.AUG_GAUSS_NOISE_VAR, p=0.3),
                A.Downscale(
                    scale_min=config.AUG_DOWNSCALE_MIN,
                    scale_max=config.AUG_DOWNSCALE_MAX, p=0.2,
                ),
            ])
        else:
            return A.Compose([
                A.MotionBlur(blur_limit=config.AUG_HEAVY_MOTION_BLUR_LIMIT, p=0.5),
                A.Perspective(scale=config.AUG_HEAVY_PERSPECTIVE_SCALE, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=config.AUG_HEAVY_BRIGHTNESS_LIMIT,
                    contrast_limit=config.AUG_HEAVY_CONTRAST_LIMIT, p=0.7,
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3),
                    shadow_dimension=5, p=0.4,
                ),
                A.GaussNoise(var_limit=config.AUG_HEAVY_GAUSS_NOISE_VAR, p=0.5),
                A.Downscale(
                    scale_min=config.AUG_HEAVY_DOWNSCALE_MIN,
                    scale_max=config.AUG_HEAVY_DOWNSCALE_MAX, p=0.4,
                ),
            ])

    def augment(self, image: np.ndarray) -> np.ndarray:
        """Applique les augmentations à une image."""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return self.transform(image=image)["image"]

    def augment_dataset(
        self, input_dir: str, output_dir: str, num_augmented_per_image: int = 3,
    ):
        """Augmente tout un dataset et sauvegarde le résultat."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
        logger.info(f"Augmentation de {len(image_files)} images ({self.mode})")

        metadata = []
        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            cv2.imwrite(str(output_path / img_file.name), image)

            for i in range(num_augmented_per_image):
                aug_image = self.augment(image)
                aug_name = f"{img_file.stem}_aug{i}{img_file.suffix}"
                cv2.imwrite(str(output_path / aug_name), aug_image)
                metadata.append({
                    "original": img_file.name, "augmented": aug_name,
                    "mode": self.mode, "index": i,
                })

        with open(output_path / "augmentation_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Augmentation terminée : {len(metadata)} images → {output_path}")


def create_train_val_test_splits(
    dataset_dir: str, output_dir: str,
    train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
    seed: int = 42,
):
    """Crée les splits train/val/test stratifiés pour IAM."""
    from sklearn.model_selection import train_test_split

    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(dataset_path.glob("*.png")) + list(dataset_path.glob("*.jpg"))
    )

    trainval, test = train_test_split(image_files, test_size=test_ratio, random_state=seed)
    val_adjusted = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(trainval, test_size=val_adjusted, random_state=seed)

    splits = {
        "train": [str(f.name) for f in train],
        "val": [str(f.name) for f in val],
        "test": [str(f.name) for f in test],
    }

    for split_name, files in splits.items():
        with open(output_path / f"{split_name}.json", "w") as f:
            json.dump(files, f, indent=2)

    logger.info(f"Splits : train={len(train)}, val={len(val)}, test={len(test)}")
    return splits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["light", "heavy"], default="light")
    parser.add_argument("--num-aug", type=int, default=3)
    args = parser.parse_args()
    augmentor = WhiteboardAugmentor(mode=args.mode)
    augmentor.augment_dataset(args.input, args.output, args.num_aug)
