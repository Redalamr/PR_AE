# 🖊️ Pipeline OCR Tableau Blanc — V0

> **Projet Majeur IA · ESAIP · IR4-S8 · 2025–2026**

Pipeline fonctionnel de bout en bout : une image de tableau blanc entre, un PDF structuré sort.

## 🚀 Installation

```bash
# Cloner le repo
git clone <url_du_repo>
cd AC_ES

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Installer Tesseract (requis pour la baseline)
# Windows : https://github.com/UB-Mannheim/tesseract/wiki
# Linux : sudo apt install tesseract-ocr
```

## 📋 Usage

```bash
# Mode webcam live (appuyer S pour capturer, Q pour quitter)
python main.py

# Image statique
python main.py --image chemin/vers/photo.jpg

# Mode batch (toutes les images d'un dossier)
python main.py --dir chemin/vers/dossier/

# Options avancées
python main.py --image photo.jpg --cnn          # Classifieur CNN (si modèle entraîné)
python main.py --image photo.jpg --doctr        # OCR docTR fine-tuné
python main.py --image photo.jpg --no-perspective   # Sans correction perspective
python main.py --image photo.jpg --binarization otsu  # Changer la binarisation
python main.py -v                                # Mode verbose
```

## 🏗️ Architecture

```
Touche clavier [S]
        │
        ▼
Capture image (webcam ou image statique)
        │
        ▼
Prétraitement image
  ├── Détection 4 coins du tableau
  ├── Correction perspective
  ├── Normalisation (CLAHE)
  └── Binarisation + débruitage
        │
        ▼
Détection des blocs (contours OpenCV)
        │
        ▼
ML #1 — Classifieur blocs (texte vs figure)
        │
        ├──► Blocs texte → ML #2 — OCR
        │
        └──► Blocs figure → recadrage PNG
        │
        ▼
Génération PDF (ReportLab)
```

## 📁 Structure du Projet

| Dossier | Description |
|---|---|
| `capture/` | Capture webcam / image statique |
| `preprocessing/` | Prétraitement (perspective, CLAHE, binarisation) |
| `layout/` | Détection et classification des blocs |
| `data/` | Gestion des données, augmentation whiteboard-style |
| `ocr/` | Moteurs OCR (Tesseract, TrOCR, docTR) |
| `output/` | Génération PDF (ReportLab) |
| `evaluation/` | Métriques (CER, WER, F1, ROC) |
| `tests/` | Tests d'intégration |
| `models/` | Modèles entraînés (.pth) |
| `notebooks/` | Notebooks d'analyse et d'ablation |

## 🧪 Tests

```bash
# Tests d'intégration
python -m tests.test_integration

# Test module par module
python -m capture.keyboard_trigger --static test_image.jpg
python -m preprocessing.pipeline test_image.jpg
python -m layout.block_detector test_image.jpg
```

## 📊 Entraînement

```bash
# Générer le dataset synthétique
python -m data.synthetic_generator --output data/synthetic --num-text 1000 --num-figure 1000

# Augmentation whiteboard-style
python -m data.augmentation_whiteboard --input data/iam --output data/iam_augmented_light --mode light
python -m data.augmentation_whiteboard --input data/iam --output data/iam_augmented_heavy --mode heavy

# Entraîner le classifieur de blocs
python -m layout.cnn_classifier --train-dir data/synthetic --val-dir data/synthetic_val

# Fine-tuning docTR (Runs A-E)
python -m ocr.training.train_doctr --run A --train-dir data/iam --val-dir data/iam_val
python -m ocr.training.train_doctr --run B --train-dir data/iam --val-dir data/iam_val
python -m ocr.training.train_doctr --run C --train-dir data/iam_augmented_light --val-dir data/iam_val
python -m ocr.training.train_doctr --run D --train-dir data/iam_augmented_heavy --val-dir data/iam_val
python -m ocr.training.train_doctr --run E --train-dir data/iam_augmented_heavy --val-dir data/iam_val
```

## 🎯 Métriques Cibles

| Métrique | Composant | Cible |
|---|---|---|
| CER | docTR fine-tuné (IAM test) | < 15% |
| WER | docTR fine-tuné (IAM test) | < 20% |
| Δ CER | Gain fine-tuning vs TrOCR brut | > 3 points |
| F1-score | Classifieur blocs CNN | > 0.80 |
| Temps pipeline | Bout en bout | < 60 secondes |

## 👥 Équipe

- **Membre 1** : Capture + Intégration pipeline
- **Membre 2** : Prétraitement image
- **Membre 3** : ML #1 — Classifieur de blocs
- **Membre 4** : Augmentation whiteboard-style
- **Membre 5** : ML #2 — Fine-tuning docTR + Comparaison OCR
- **Membre 6** : Évaluation + PDF + Tests intégration
