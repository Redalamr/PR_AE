"""
Script d'entraînement docTR — Fine-tuning sur IAM.

5 configurations (ablation study) :
- Run A : Freeze encoder, train decoder uniquement
- Run B : Full fine-tuning, IAM seul, sans augmentation
- Run C : Full fine-tuning, IAM + augmentation whiteboard légère
- Run D : Full fine-tuning, IAM + augmentation whiteboard agressive
- Run E : Full fine-tuning, IAM + augmentation agressive + curriculum learning

Stack : PyTorch, docTR, MLflow
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Optional
import logging
import mlflow

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

logger = logging.getLogger(__name__)


class IAMDocTRDataset(Dataset):
    """Dataset IAM formaté pour docTR fine-tuning."""

    def __init__(self, data_dir: str, split_file: Optional[str] = None, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.entries = []

        if split_file and Path(split_file).exists():
            with open(split_file, "r") as f:
                file_list = json.load(f)
            for fname in file_list:
                img_path = self.data_dir / fname
                label_path = img_path.with_suffix(".txt")
                if img_path.exists() and label_path.exists():
                    self.entries.append({
                        "image_path": str(img_path),
                        "label": label_path.read_text().strip(),
                    })
        else:
            for img_file in sorted(self.data_dir.glob("*.png")):
                label_path = img_file.with_suffix(".txt")
                if label_path.exists():
                    self.entries.append({
                        "image_path": str(img_file),
                        "label": label_path.read_text().strip(),
                    })
        logger.info(f"IAMDocTRDataset : {len(self.entries)} entrées")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = cv2.imread(entry["image_path"])
        if image is None:
            raise ValueError(f"Cannot read: {entry['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            result = self.transform(image=image)
            image = result["image"]
        return {"image": image, "label": entry["label"]}


def train_doctr(
    run_name: str, train_dir: str, val_dir: str,
    train_split: Optional[str] = None, val_split: Optional[str] = None,
    freeze_encoder: bool = False,
    epochs: int = config.DOCTR_TRAIN_EPOCHS,
    batch_size: int = config.DOCTR_TRAIN_BATCH_SIZE,
    lr: float = config.DOCTR_TRAIN_LR,
    patience: int = config.DOCTR_TRAIN_PATIENCE,
    curriculum: bool = False, transform=None,
    save_path: Optional[Path] = None,
):
    """Entraîne le modèle docTR (recognizer CRNN)."""
    from doctr.models import recognition

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"=== Run {run_name} === Device: {device}")

    model = recognition.crnn_vgg16_bn(pretrained=True).to(device)

    if freeze_encoder:
        for param in model.feat_extractor.parameters():
            param.requires_grad = False
        logger.info("Encoder gelé (Run A)")

    train_dataset = IAMDocTRDataset(train_dir, train_split, transform)
    val_dataset = IAMDocTRDataset(val_dir, val_split)

    if curriculum:
        train_dataset.entries.sort(key=lambda e: len(e["label"]))
        logger.info("Curriculum learning activé")

    # Fix: DataLoader batch collation custom function
    def collate_fn(batch):
        images = [cv2.resize(b["image"], (128, 32)) for b in batch]  # taille fixe docTR
        images = [torch.from_numpy(img.transpose(2,0,1)).float() / 255.0 for img in images]
        labels = [b["label"] for b in batch]
        return {"image": torch.stack(images), "label": labels}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=not curriculum, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    criterion = nn.CTCLoss(zero_infinity=True)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_OCR)
    save_path = save_path or (config.MODELS_DIR / "doctr_finetuned_best.pth")

    with mlflow.start_run(run_name=f"doctr_{run_name}"):
        mlflow.log_params({
            "run_name": run_name, "freeze_encoder": freeze_encoder,
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "curriculum": curriculum,
            "train_size": len(train_dataset), "val_size": len(val_dataset),
        })

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            num_batches = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Conversion des données pour docTR
                # Selon les transforms, images peut être un tenseur ou une liste
                if isinstance(batch["image"], torch.Tensor):
                    images = batch["image"].to(device)
                else:
                    images = [img.to(device) for img in batch["image"]]
                labels = batch["label"]
                
                # Forward pass qui calcule la loss (CTC) en interne si les cibles sont fournies
                out = model(images, target=labels)
                loss = out["loss"]
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1

            train_loss /= max(1, num_batches)

            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch["image"], torch.Tensor):
                        images = batch["image"].to(device)
                    else:
                        images = [img.to(device) for img in batch["image"]]
                    labels = batch["label"]
                    
                    out = model(images, target=labels)
                    val_loss += out["loss"].item()
                    val_batches += 1
                    
            val_loss /= max(1, val_batches)

            logger.info(f"Epoch [{epoch+1}/{epochs}] — Train: {train_loss:.4f}, Val: {val_loss:.4f}")
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                logger.info(f"  → Meilleur modèle sauvegardé")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping à l'époque {epoch+1}")
                    break

        mlflow.log_metric("best_val_loss", best_val_loss)
    logger.info(f"Run {run_name} terminé — best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tuning docTR")
    parser.add_argument("--run", required=True, choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--epochs", type=int, default=config.DOCTR_TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DOCTR_TRAIN_BATCH_SIZE)
    args = parser.parse_args()

    run_configs = {
        "A": {"freeze_encoder": True, "curriculum": False},
        "B": {"freeze_encoder": False, "curriculum": False},
        "C": {"freeze_encoder": False, "curriculum": False},
        "D": {"freeze_encoder": False, "curriculum": False},
        "E": {"freeze_encoder": False, "curriculum": True},
    }
    cfg = run_configs[args.run]

    transform = None
    if args.run in ("C", "D"):
        from data.augmentation_whiteboard import WhiteboardAugmentor
        mode = "light" if args.run == "C" else "heavy"
        augmentor = WhiteboardAugmentor(mode=mode)
        transform = augmentor.transform

    train_doctr(
        run_name=args.run, train_dir=args.train_dir, val_dir=args.val_dir,
        train_split=args.train_split, val_split=args.val_split,
        freeze_encoder=cfg["freeze_encoder"], epochs=args.epochs,
        batch_size=args.batch_size, curriculum=cfg["curriculum"], transform=transform,
    )
