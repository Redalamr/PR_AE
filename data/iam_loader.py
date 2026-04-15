"""
Chargement et préparation du dataset IAM Handwriting Database.
IAM : 115 000 lignes manuscrites anglaises, 657 scripteurs.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from torch.utils.data import Dataset
import logging

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


class IAMDataset(Dataset):
    """Dataset PyTorch pour le chargement des images IAM."""

    def __init__(
        self, data_dir: str, split_file: Optional[str] = None,
        transform=None, max_label_length: int = 100,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_label_length = max_label_length
        self.entries = self._load_entries(split_file)
        logger.info(f"IAMDataset : {len(self.entries)} images")

    def _load_entries(self, split_file: Optional[str]) -> List[Dict]:
        """Charge la liste des images et labels."""
        entries = []
        if split_file and Path(split_file).exists():
            with open(split_file, "r") as f:
                file_list = json.load(f)
            for fname in file_list:
                img_path = self.data_dir / fname
                label_path = img_path.with_suffix(".txt")
                if img_path.exists():
                    label = label_path.read_text().strip() if label_path.exists() else ""
                    entries.append({"image_path": str(img_path), "label": label})
        else:
            for img_file in sorted(self.data_dir.glob("*.png")):
                label_path = img_file.with_suffix(".txt")
                label = label_path.read_text().strip() if label_path.exists() else ""
                entries.append({"image_path": str(img_file), "label": label})
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.entries[idx]
        image = cv2.imread(entry["image_path"])
        if image is None:
            raise ValueError(f"Cannot read: {entry['image_path']}")
        if self.transform:
            try:
                result = self.transform(image=image)
                image = result["image"]
            except Exception:
                pass
        return {
            "image": image,
            "label": entry["label"][:self.max_label_length],
            "path": entry["image_path"],
        }

    def get_stats(self) -> Dict:
        """Statistiques descriptives du dataset."""
        labels = [e["label"] for e in self.entries]
        lengths = [len(l) for l in labels]
        vocab = set()
        for label in labels:
            vocab.update(label)
        return {
            "num_images": len(self.entries),
            "avg_label_length": np.mean(lengths) if lengths else 0,
            "max_label_length": max(lengths) if lengths else 0,
            "min_label_length": min(lengths) if lengths else 0,
            "vocab_size": len(vocab),
            "vocab": sorted(vocab),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-file", default=None)
    args = parser.parse_args()
    dataset = IAMDataset(args.data_dir, args.split_file)
    stats = dataset.get_stats()
    print(f"IAM : {stats['num_images']} images, vocab={stats['vocab_size']} chars")
