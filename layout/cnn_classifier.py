"""
Étape 4 — ML #1 : Classifieur CNN de blocs (texte vs figure).

Architecture : MobileNetV2 (pré-entraîné ImageNet, tête remplacée)
Classes : texte, figure
Augmentation : rotation ±15°, flips, jitter luminosité, bruit gaussien
Optimiseur : Adam lr=1e-4

Stack : PyTorch + torchvision
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import Tuple, List, Optional
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


class CNNBlockClassifier:
    """
    Classifieur CNN binaire : texte vs figure.
    MobileNetV2 pré-entraîné avec tête de classification remplacée.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or config.CLASSIFIER_MODEL_PATH

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.CLASSIFIER_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.CLASSIFIER_INPUT_SIZE),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.model = self._build_model()
        self._load_weights()

    def _build_model(self) -> nn.Module:
        """Construit MobileNetV2 avec tête remplacée."""
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, config.CLASSIFIER_NUM_CLASSES),
        )
        return model.to(self.device)

    def _load_weights(self):
        """Charge les poids du modèle si disponibles."""
        if self.model_path.exists():
            state_dict = torch.load(
                self.model_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict)
            logger.info(f"Poids chargés depuis {self.model_path}")
        else:
            logger.warning(f"Poids non trouvés ({self.model_path}). Modèle ImageNet.")

    def classify(self, block_image: np.ndarray) -> Tuple[str, float]:
        """
        Classifie un bloc en 'text' ou 'figure'.

        Args:
            block_image: Image du bloc (H, W) ou (H, W, 3) uint8.

        Returns:
            Tuple (label, confidence).
        """
        self.model.eval()

        if len(block_image.shape) == 2:
            block_image = cv2.cvtColor(block_image, cv2.COLOR_GRAY2BGR)

        tensor = self.transform(block_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        label = config.BLOCK_CLASSES[predicted.item()]
        return (label, confidence.item())

    def classify_batch(self, block_images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Classifie une liste de blocs."""
        return [self.classify(img) for img in block_images]

    def save_model(self, path: Optional[Path] = None):
        """Sauvegarde les poids du modèle."""
        save_path = path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Modèle sauvegardé → {save_path}")


def train_classifier(
    train_dir: str,
    val_dir: str,
    epochs: int = config.CLASSIFIER_EPOCHS,
    batch_size: int = config.CLASSIFIER_BATCH_SIZE,
    lr: float = config.CLASSIFIER_LR,
    save_path: Optional[Path] = None,
):
    """
    Entraîne le classifieur CNN sur les patches texte/figure.

    Args:
        train_dir: Dossier d'entraînement avec sous-dossiers 'text/' et 'figure/'
        val_dir: Dossier de validation même structure.
    """
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    import mlflow

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Entraînement sur {device}")

    train_transform = transforms.Compose([
        transforms.Resize(config.CLASSIFIER_INPUT_SIZE),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(config.CLASSIFIER_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    logger.info(f"Train : {len(train_dataset)}, Val : {len(val_dataset)}")

    classifier = CNNBlockClassifier(
        model_path=save_path or config.CLASSIFIER_MODEL_PATH, device=device,
    )
    model = classifier.model
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_CLASSIFIER)

    with mlflow.start_run(run_name="cnn_block_classifier"):
        mlflow.log_params({
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "architecture": "MobileNetV2",
            "train_size": len(train_dataset), "val_size": len(val_dataset),
        })

        best_val_loss = float("inf")

        for epoch in range(epochs):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_loss /= train_total
            train_acc = train_correct / train_total

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= val_total
            val_acc = val_correct / val_total
            scheduler.step(val_loss)

            logger.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f}"
            )
            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
            }, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                classifier.save_model(save_path)
                logger.info(f"  → Meilleur modèle (val_loss={val_loss:.4f})")

        mlflow.log_metric("best_val_loss", best_val_loss)
    logger.info("Entraînement terminé")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--epochs", type=int, default=config.CLASSIFIER_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.CLASSIFIER_BATCH_SIZE)
    args = parser.parse_args()
    train_classifier(args.train_dir, args.val_dir, args.epochs, args.batch_size)
