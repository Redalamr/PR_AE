# Plan V0 — Pipeline Complet (sans MediaPipe)
## Projet Majeur IA · ESAIP · IR4-S8 · 2025–2026

> **Objectif V0 :** un pipeline fonctionnel de bout en bout — une image de tableau blanc entre, un PDF structuré sort. Tout le ML est entraîné et évalué. Seul le déclenchement par geste (MediaPipe) est remplacé par une touche clavier.

> **V1 (après V0) :** uniquement remplacer la touche clavier par MediaPipe + rapport + soutenance.

---

## Pipeline V0 — Vue d'ensemble

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
        ├──► Blocs texte → ML #2 — OCR docTR fine-tuné
        │
        └──► Blocs figure → recadrage PNG
        │
        ▼
Génération PDF (ReportLab)
```

---

## Étape 1 — Capture (touche clavier)

**Ce que ça fait :** appui sur `S` → capture la frame courante de la webcam et la sauvegarde.

**Stack :** OpenCV `VideoCapture` + `cv2.imwrite()`

**Dataset :** aucun — c'est du code pur.

**Livrable :** `capture/keyboard_trigger.py` — fonctionnel, testé sur images statiques et webcam live.

---

## Étape 2 — Prétraitement image (complet)

**Ce que ça fait :**
1. Détection automatique des 4 coins du tableau (Canny + `approxPolyDP`)
2. Correction de perspective (`findHomography` + `warpPerspective`)
3. Normalisation photométrique (CLAHE adaptatif)
4. Binarisation (Otsu global + seuillage adaptatif local — comparaison des deux)
5. Débruitage morphologique (ouverture + fermeture)

**Stack :** OpenCV, Pillow

**Dataset :** aucun — traitement classique.

**Ablation study prétraitement :** comparer 4 configurations (Otsu seul / adaptatif seul / CLAHE + Otsu / CLAHE + adaptatif) sur 30 images tests → mesurer impact sur CER final. Résultat inclus dans le rapport.

**Livrable :** `preprocessing/pipeline.py` + notebook `preprocessing_ablation.ipynb`

---

## Étape 3 — Détection des blocs (heuristique OpenCV)

**Ce que ça fait :** à partir de l'image binarisée, trouve des régions candidates (bounding boxes) qui seront ensuite envoyées au classifieur ML #1.

**Stack :** OpenCV — projection horizontale + `findContours` + filtrage par taille

**Dataset :** aucun — heuristique pure.

**Livrable :** `layout/block_detector.py` — retourne une liste de bounding boxes triées haut→bas, gauche→droite.

---

## Étape 4 — ML #1 : Classifieur de blocs (texte vs figure)

**Ce que ça fait :** pour chaque bounding box détectée, classe le contenu en `texte` ou `figure`.

### Dataset utilisé : synthétique + DocLayNet

**Pourquoi synthétique :**
On génère programmatiquement des patches simulant du texte (lignes horizontales, densité élevée, structure régulière) et des figures (cercles, courbes, flèches, graphes simples) avec OpenCV. Reproductible, gratuit, volume illimité.

**Source complémentaire : DocLayNet (IBM, open-source)**
Dataset de layout documentaire annoté (texte / figure / tableau / titre). On extrait uniquement les patches texte et figure pour notre classifieur binaire. ~80 000 exemples disponibles, licence permissive.

**Pipeline entraînement :**
- Architecture : MobileNetV2 (pré-entraîné ImageNet, tête remplacée)
- Classes : `texte`, `figure`
- Dataset entraînement : patches synthétiques + extraits DocLayNet
- Augmentation : rotation ±15°, flips, jitter luminosité, bruit gaussien
- Optimiseur : Adam lr=1e-4
- Métriques : F1-score, Precision, Recall, matrice de confusion

**Comparaison (exigence académique) :**
CNN fine-tuné vs heuristique baseline (ratio d'aspect + densité de pixels sombres)

**Stack :** PyTorch + torchvision, MLflow (tracking)

**Livrables :** `layout/cnn_classifier.py` + `models/block_classifier.pth` + notebook `block_classifier_eval.ipynb`

---

## Étape 5 — ML #2 : OCR — Fine-tuning docTR

**Ce que ça fait :** reconnaître le texte manuscrit anglais dans chaque bloc texte détecté.

### Dataset utilisé

| Dataset | Description | Usage |
|---|---|---|
| **IAM Handwriting Database** | 115 000 lignes manuscrites anglaises, 657 scripteurs, benchmark standard HTR | Base principale de fine-tuning |
| **IAM augmenté (whiteboard-style)** | IAM transformé via pipeline d'augmentation synthétique simulant les conditions tableau blanc (flou, distorsion, éclairage inégal, bruit) | Adaptation domaine sans collecte manuelle |

**Justification académique de l'approche :**
L'adaptation de domaine par augmentation synthétique est une stratégie standard en HTR lorsque la collecte manuelle n'est pas possible. Le pipeline d'augmentation simule fidèlement les dégradations propres aux tableaux blancs photographiés (reflets, flou de mouvement, perspective résiduelle, contraste faible). Cette approche est documentée dans la littérature (SynthText, TextRecognitionDataGenerator).

### Pipeline augmentation whiteboard-style

Transformations appliquées sur les images IAM pour simuler les conditions tableau blanc :
- **Flou de mouvement** (`albumentations.MotionBlur`) — simule bougé caméra
- **Distorsion de perspective légère** (`albumentations.Perspective`) — inclinaison résiduelle du tableau
- **Variations d'éclairage** (`albumentations.RandomBrightnessContrast`, `RandomShadow`) — reflets et zones sombres
- **Bruit gaussien** (`albumentations.GaussNoise`) — grain numérique webcam
- **Dégradation de résolution** (`albumentations.Downscale`) — simulation webcam bas de gamme

**Modèle :** docTR (Mindee, HuggingFace) — DBNet détection + CRNN recognizer intégré, fine-tunable

**Ablation study — 5 configurations comparées :**

| Run | Config | Description |
|---|---|---|
| A | Freeze encoder, train decoder | Adaptation rapide sans coût computationnel élevé |
| B | Full fine-tuning, IAM seul, sans augmentation | Référence baseline pure |
| C | Full fine-tuning, IAM + augmentation whiteboard légère | Simulation domaine minimale |
| D | Full fine-tuning, IAM + augmentation whiteboard agressive | Simulation domaine complète |
| E | Full fine-tuning, IAM + augmentation agressive + curriculum learning | Impact de l'ordre de présentation des exemples |

**Comparaison 3 moteurs OCR :**

| Moteur | Type | Entraînement |
|---|---|---|
| Tesseract 5 (`eng`) | Classique | Aucun — off-the-shelf |
| TrOCR (HuggingFace) | DL | Aucun — pré-entraîné anglais |
| docTR fine-tuné | DL | Fine-tuné sur IAM + IAM augmenté whiteboard-style |

**Métriques :** CER (Character Error Rate) + WER (Word Error Rate) sur :
- IAM test set officiel (comparaison avec état de l'art)
- Sous-ensemble IAM augmenté mis de côté dès le début (validation adaptation domaine)

**Stack :** PyTorch, HuggingFace Trainer, albumentations, MLflow, Google Colab Pro (GPU)

**Livrables :** `ocr/doctr_engine.py` + `ocr/training/train_doctr.py` + `models/doctr_finetuned_best.pth` + notebook `ocr_ablation_results.ipynb`

---

## Étape 6 — Génération PDF

**Ce que ça fait :** assemble le texte reconnu + les figures recadrées dans un PDF structuré et indexable (searchable), en préservant l'ordre spatial du tableau.

**Stack :** ReportLab

**Livrable :** `output/pdf_generator.py` — retourne un fichier `.pdf` horodaté.

---

## Métriques cibles V0

| Métrique | Composant | Cible |
|---|---|---|
| CER | docTR fine-tuné (IAM test set officiel) | < 15% |
| WER | docTR fine-tuné (IAM test set officiel) | < 20% |
| Δ CER | Gain fine-tuning + augmentation vs TrOCR brut | > 3 points |
| Δ CER augmentation | Gain Run D vs Run B (impact augmentation whiteboard) | mesurable et positif |
| F1-score | Classifieur blocs CNN | > 0.80 |
| Gain F1 | CNN vs heuristique | mesurable |
| Temps pipeline | Bout en bout | < 60 secondes |

---

## Ce que la V0 ne contient PAS (reporté en V1)

- Déclenchement MediaPipe (zone SAVE + détection doigt)
- Collecte manuelle de données propriétaires
- Rapport final
- Slides et soutenance
