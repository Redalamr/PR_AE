# Division des Tâches V0 — 6 Membres
## Projet Majeur IA · ESAIP · IR4-S8 · 2025–2026

> **Total :** 6 × 25h = 150h pour la V0  
> **Les 30h restantes** (6 × 5h) sont réservées pour la V1 (MediaPipe) + rapport + soutenance

---

## Vue d'ensemble

| Membre | Module | Heures V0 |
|---|---|---|
| Membre 1 | Capture clavier + Intégration pipeline | 25h |
| Membre 2 | Prétraitement image | 25h |
| Membre 3 | ML #1 — Classifieur de blocs | 25h |
| Membre 4 | Augmentation whiteboard-style + Support intégration | 25h |
| Membre 5 | ML #2 — Fine-tuning docTR + Comparaison OCR | 25h |
| Membre 6 | Évaluation + Génération PDF + Tests intégration | 25h |

---

## Membre 1 — Capture clavier + Intégration pipeline

**Rôle :** point d'entrée du pipeline + responsable que tous les modules se connectent

| # | Tâche | Heures |
|---|---|---|
| 1.1 | Setup repo GitHub : structure dossiers, branches par module, README principal | 2h |
| 1.2 | Module capture clavier : touche `S` → capture webcam → sauvegarde image horodatée | 3h |
| 1.3 | Définir et documenter les interfaces entre modules (contrats Python) en semaine 1 | 2h |
| 1.4 | Script `main.py` qui orchestre tout le pipeline bout en bout | 4h |
| 1.5 | Tests d'intégration : pipeline complet sur 10 images statiques calibrées | 4h |
| 1.6 | Gestion des cas d'erreur : image floue, tableau non détecté, OCR vide | 3h |
| 1.7 | Coordination équipe : suivi GitHub Projects, réunions hebdo | 4h |
| 1.8 | Tests finaux V0 sur tableau blanc réel (3 conditions d'éclairage) | 3h |
| **Total** | | **25h** |

**Livrables :**
- `capture/keyboard_trigger.py`
- `main.py` (pipeline complet)
- `tests/test_integration.py`
- `README.md` (instructions d'installation et d'exécution)

---

## Membre 2 — Prétraitement image

**Rôle :** transformer l'image brute capturée en image propre, redressée, binarisée

| # | Tâche | Heures |
|---|---|---|
| 2.1 | Détection automatique des 4 coins du tableau (Canny + approxPolyDP + filtrage) | 5h |
| 2.2 | Correction de perspective (findHomography + warpPerspective) | 4h |
| 2.3 | Normalisation photométrique : CLAHE adaptatif | 3h |
| 2.4 | Implémentation et comparaison binarisation Otsu vs adaptative locale | 3h |
| 2.5 | Débruitage morphologique (ouverture, fermeture) | 2h |
| 2.6 | Ablation study prétraitement : 4 configurations comparées sur 30 images, impact sur CER mesuré | 5h |
| 2.7 | Fallback : calibration manuelle des corners si détection automatique échoue | 2h |
| 2.8 | Tests unitaires + mesure temps d'exécution (< 2 secondes) | 1h |
| **Total** | | **25h** |

**Livrables :**
- `preprocessing/perspective.py`
- `preprocessing/enhance.py`
- `preprocessing/pipeline.py` (façade unique — interface pour Membre 3)
- `notebooks/preprocessing_ablation.ipynb`

**Interface exposée :**
```python
class PreprocessingPipeline:
    def run(self, raw_frame: np.ndarray) -> np.ndarray:
        # retourne image binarisée redressée (H, W) uint8
```

---

## Membre 3 — ML #1 : Classifieur de blocs

**Rôle :** détecter les régions du tableau et les classifier en texte ou figure

| # | Tâche | Heures |
|---|---|---|
| 3.1 | Détection des blocs candidats (contours OpenCV + bounding boxes + tri spatial) | 3h |
| 3.2 | Baseline heuristique : ratio d'aspect + densité pixels sombres + Hu moments | 3h |
| 3.3 | Génération dataset synthétique : patches texte (lignes horizontales régulières) et figures (cercles, courbes, flèches) avec OpenCV | 4h |
| 3.4 | Extraction patches texte/figure depuis DocLayNet (IBM, open-source) | 2h |
| 3.5 | Setup pipeline entraînement CNN (MobileNetV2, DataLoader PyTorch, augmentation) | 3h |
| 3.6 | Fine-tuning MobileNetV2 sur dataset synthétique + DocLayNet | 3h |
| 3.7 | Évaluation : F1, précision, rappel, matrice de confusion — CNN vs heuristique | 3h |
| 3.8 | Analyse des erreurs (quels types de blocs mal classifiés) | 2h |
| 3.9 | Tests unitaires classifieur | 2h |
| **Total** | | **25h** |

**Livrables :**
- `layout/block_detector.py` (détection contours)
- `layout/heuristic_classifier.py`
- `layout/cnn_classifier.py` + `models/block_classifier.pth`
- `notebooks/block_classifier_eval.ipynb`

**Interface exposée :**
```python
class BlockDetector:
    def detect_and_classify(self, binary_image: np.ndarray) -> List[Block]:
        # retourne liste de Block(bbox, label, confidence)
        # label ∈ {'text', 'figure'}
```

---

## Membre 4 — Augmentation whiteboard-style + Support intégration

**Rôle :** construire le pipeline d'augmentation synthétique qui permet à docTR de s'adapter au domaine tableau blanc sans collecte manuelle, et soutenir l'intégration du pipeline

> **Justification :** en lieu et place d'une collecte manuelle coûteuse en temps, on applique des transformations ciblées sur IAM pour simuler fidèlement les dégradations propres aux tableaux blancs photographiés (flou, reflets, perspective, faible contraste). Cette stratégie — dite *synthetic domain adaptation* — est académiquement justifiable et documentée dans la littérature HTR.

| # | Tâche | Heures |
|---|---|---|
| 4.1 | Étude des dégradations spécifiques aux tableaux blancs (littérature + analyse visuelle) | 2h |
| 4.2 | Implémentation des transforms albumentations : flou de mouvement, distorsion perspective, variations d'éclairage, bruit gaussien, dégradation de résolution | 6h |
| 4.3 | Calibration des hyperparamètres d'augmentation (intensité légère vs agressive) pour les runs C et D de docTR | 3h |
| 4.4 | Génération et export des deux versions augmentées d'IAM (légère + agressive) au format HuggingFace compatible docTR | 3h |
| 4.5 | Validation visuelle : inspection manuelle d'un échantillon de 100 images augmentées pour confirmer le réalisme | 2h |
| 4.6 | Split train/val/test IAM — stratifié, test set mis de côté immédiatement | 2h |
| 4.7 | Stats descriptives dataset IAM (nb scripteurs, distribution longueurs de ligne, vocabulaire) | 1h |
| 4.8 | Tests unitaires du pipeline d'augmentation (reproductibilité, format de sortie) | 2h |
| 4.9 | Support intégration Membre 1 : aide au câblage pipeline complet + debug S5–S9 | 4h |
| **Total** | | **25h** |

**Livrables :**
- `data/augmentation_whiteboard.py` (pipeline albumentations complet)
- `data/iam_augmented_light/` — version augmentation légère (Run C)
- `data/iam_augmented_heavy/` — version augmentation agressive (Run D & E)
- `data/splits/` : `train.json` / `val.json` / `test.json`
- `data/README_augmentation.md` : documentation des transforms, paramètres, justification académique

> ⚠️ Le test set IAM est sous la responsabilité exclusive de Membre 4. Aucun autre membre n'y accède avant la phase d'évaluation finale.

---

## Membre 5 — ML #2 : Fine-tuning docTR + Comparaison OCR

**Rôle :** entraîner le modèle OCR principal et comparer les 3 approches

| # | Tâche | Heures |
|---|---|---|
| 5.1 | Setup Tesseract 5 mode `eng` + intégration Python (baseline 1) | 2h |
| 5.2 | Setup TrOCR HuggingFace off-the-shelf (baseline 2) | 2h |
| 5.3 | Setup docTR + compréhension architecture DBNet + CRNN | 3h |
| 5.4 | Setup MLflow (tracking local : hyperparamètres, métriques, artefacts) | 1h |
| 5.5 | **Run A :** freeze encoder, train decoder uniquement (IAM seul) | 2h |
| 5.6 | **Run B :** full fine-tuning, IAM seul, sans augmentation (référence baseline) | 2h |
| 5.7 | **Run C :** full fine-tuning, IAM + augmentation whiteboard légère (fournie par M4) | 2h |
| 5.8 | **Run D :** full fine-tuning, IAM + augmentation whiteboard agressive (fournie par M4) | 2h |
| 5.9 | **Run E :** full fine-tuning, IAM + augmentation agressive + curriculum learning | 3h |
| 5.10 | Analyse résultats : courbes de loss, CER/WER par run, sélection meilleur modèle | 3h |
| 5.11 | Tests unitaires moteurs OCR (même input → comparer outputs) | 3h |
| **Total** | | **25h** |

**Datasets utilisés :**
- IAM Handwriting Database (anglais, 115 000 lignes) — base fine-tuning, fourni par Membre 4
- IAM augmenté whiteboard-style légère + agressive — adaptation domaine synthétique, fourni par Membre 4

**Livrables :**
- `ocr/tesseract_engine.py`
- `ocr/trocr_engine.py`
- `ocr/doctr_engine.py` + `ocr/training/train_doctr.py`
- `models/doctr_finetuned_best.pth`
- `notebooks/ocr_ablation_results.ipynb`

**Interface exposée :**
```python
class OCREngine:
    def recognize(self, block_image: np.ndarray) -> OCRResult:
        # retourne OCRResult(text: str, confidence: float)
```

---

## Membre 6 — Évaluation + PDF + Tests intégration

**Rôle :** mesurer les performances, générer le PDF final, et s'assurer que tout marche ensemble

| # | Tâche | Heures |
|---|---|---|
| 6.1 | Pipeline évaluation CER/WER (jiwer) sur les 3 moteurs × 2 test sets (IAM + IAM augmenté) | 4h |
| 6.2 | Pipeline évaluation classifieur (F1, précision, rappel, matrice de confusion, ROC) | 3h |
| 6.3 | Analyse de l'impact de l'augmentation : Δ CER Run B vs Run C vs Run D | 2h |
| 6.4 | Génération figures et tableaux de résultats (matplotlib/seaborn) | 3h |
| 6.5 | Module génération PDF (ReportLab) : texte + figures PNG, ordre spatial préservé | 5h |
| 6.6 | Tests intégration bout en bout sur 10 images réelles standardisées | 4h |
| 6.7 | Mesure latence par module + temps pipeline total | 2h |
| 6.8 | Dashboard résultats notebook | 2h |
| **Total** | | **25h** |

**Livrables :**
- `output/pdf_generator.py`
- `evaluation/ocr_evaluator.py`
- `evaluation/classifier_evaluator.py`
- `tests/test_integration.py` (suite complète)
- `notebooks/final_results_dashboard.ipynb`

---

## Calendrier V0 — 12 semaines

```
S1   Setup repo + interfaces contractuelles définies + M4 : étude dégradations + début augmentation
S2   M1: capture clavier · M2: détection corners · M4: implémentation transforms albumentations
S3   M2: binarisation complète · M3: dataset synthétique · M4: génération IAM augmenté léger + agressif
S4   M2: ablation prétraitement · M3: DocLayNet + baseline heuristique · M4: validation visuelle + splits
     ── MILESTONE 1 : capture → prétraitement opérationnel + datasets augmentés prêts ──
S5   M3: setup CNN + entraînement · M5: setup Tesseract + TrOCR + docTR
S6   M3: fine-tuning classifieur · M5: Runs A + B · M6: pipeline évaluation CER/WER
S7   M3: évaluation classifieur · M5: Runs C + D · M6: pipeline évaluation F1
S8   M5: Run E + sélection meilleur modèle · M6: analyse impact augmentation
     ── MILESTONE 2 : tous les runs ML terminés ──
S9   M6: tests intégration end-to-end · M1: debug pipeline complet · M4: support intégration
S10  M6: génération PDF · M1: tests sur tableau blanc réel
S11  Corrections finales + mesure latence + freeze code V0
     ── MILESTONE 3 : V0 livrée et fonctionnelle ──
S12  Début V1 (MediaPipe) + début rapport
```

---

## Règles Git équipe

```
main          ← code stable uniquement (merge par milestone)
├── dev       ← intégration continue
│   ├── feature/capture        (M1)
│   ├── feature/preprocessing  (M2)
│   ├── feature/layout         (M3)
│   ├── feature/data           (M4)
│   ├── feature/ocr            (M5)
│   └── feature/evaluation     (M6)
```

- Merge sur `dev` → Pull Request obligatoire + 1 reviewer
- Pas de commit direct sur `main` ou `dev`
- Chaque module développé avec des mocks des autres modules → pas de blocage
- Réunion hebdomadaire 20 min (lundi) : quoi fait, quoi prévu, blocages
