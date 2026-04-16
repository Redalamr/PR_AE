# Historique des Versions et Modifications

Ce document résume l'évolution du projet de la version initiale (V1) jusqu'à la version actuelle (V3), ainsi que les modifications apportées pour chaque transition.

## Version 1 (V1) : Pipeline Initial
La première version a posé les fondations du projet.
* **Mise en place de l'application Streamlit** : Interface basique de téléchargement d'images.
* **Prétraitement basique** : Opérations d'images elementaires (redimensionnement, conversion).
* **Extraction OCR simple** : Intégration initiale pour lire le texte global sans séparation sémantique poussée.
* **Génération PDF** : Export basique des résultats textuels dans un document PDF.

---

## Version 2 (V2) : Stabilisation et Algorithmes Heuristiques (Pipeline Classique)
La V2 s'est concentrée sur l'analyse de disposition (layout) et la fiabilité des modèles OCR en local.
* **Analyse Heuristique** : Création du `HeuristicClassifier` et remplacement du détecteur initial par un **algorithme de fusion multi-passe** pour séparer efficacement les blocs de texte et les figures.
* **Modèles OCR Avancés** : Intégration stable de **Tesseract 5**, **TrOCR** (Microsoft) et **docTR** (Mindee).
* **Mode "Démo Stable"** : Mise en place de configurations sécurisées de l'interface graphique pour éviter les plantages lors des démonstrations.
* **Correction d'erreurs (OOM)** : Gestion de l'allocation mémoire pour TrOCR et autres modèles lourds.
* **Documentation** : Création de la documentation d'architecture et de mise en place.

---

## Version 3 (V3) : Intégration du Pipeline IA (Vision Models)
La V3 a marqué un tournant avec l'intégration de modèles IA de vision de pointe pour l'analyse de structure, surmontant les limites des heuristiques.
* **YOLO-World & Surya** : Remplacement de l'approche heuristique par YOLO-World (pour la détection globale des éléments de layout) et Surya (pour la détection précise des lignes de texte).
* **Modules d'API `layout/`** : Implémentation des intégrateurs API locaux (`api_yolo.py` et `api_surya.py`) intercommuniquant avec les backends IA (via Gradio / appels JSON).
* **Optimisations de Transfert** : Compression client, redimensionnement et encodage Base64 des images avant les requêtes REST pour éviter l'erreur `413 Payload Too Large`.
* **Gestion Robuste des Fichiers** : Décodage et lecture appropriée des réponses `filepath` (chemins temporaires JSON) de Surya pour extraire les prédictions finales.
* **Choix de Pipeline UI** : Modification de la sidebar Streamlit permettant à l'utilisateur de basculer librement et sans interruption entre le **Pipeline Classique (V2 - Heuristiques)** et le nouveau **Pipeline IA (V3)**.
