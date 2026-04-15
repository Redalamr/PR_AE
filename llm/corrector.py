"""
Module V2 — Correction post-OCR via LLM.

Corrige les erreurs typiques de l'OCR sur du vocabulaire technique
(IA, ML, Deep Learning) en envoyant le texte brut à un LLM
avec un prompt système très spécifique.

Supporte :
  - OpenAI (GPT-4o-mini, GPT-4o)
  - Anthropic (Claude 3.5 Sonnet, Claude 3 Haiku)
  - Mode simulation (fallback sans API key)

Usage :
    corrector = LLMCorrector(provider="openai", api_key="sk-...")
    corrected = corrector.correct("Le reseau de neuronnes utilise PyToroh")
"""

import os
import re
import logging
from typing import Optional, Literal
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────
# Prompt système — NE PAS MODIFIER SANS RAISON
# ────────────────────────────────────────────
SYSTEM_PROMPT = """Tu es un correcteur orthographique spécialisé pour des notes de cours universitaires
en Intelligence Artificielle, Machine Learning et Deep Learning.

RÈGLES STRICTES :
1. Corrige UNIQUEMENT les fautes d'orthographe, de grammaire et de vocabulaire technique.
2. Ne reformule JAMAIS les phrases. Ne réordonne pas les mots.
3. Préserve intégralement la structure originale (retours à la ligne, tirets, puces, numérotation).
4. Corrige les erreurs OCR courantes sur le vocabulaire technique :
   - "PyToroh" → "PyTorch"
   - "TensorFIow" → "TensorFlow"
   - "convolutionne1" → "convolutionnel"
   - "rétropropagatiom" → "rétropropagation"
   - "gradlent" → "gradient"
   - "descente de gradiant" → "descente de gradient"
   - "fonctlon" → "fonction"
   - "réseau de neuronnes" → "réseau de neurones"
   - "apprentlssage" → "apprentissage"
   - "classificateur" ne doit PAS être changé en "classifieur" (les deux sont valides)
   - "epoch" et "époque" sont tous les deux valides
5. Préserve tous les termes anglais techniques tels quels : "batch size", "learning rate",
   "dropout", "softmax", "cross-entropy", "backpropagation", "overfitting", "underfitting".
6. Préserve les formules LaTeX ou mathématiques telles quelles (ne modifie rien entre $...$ ou $$...$$).
7. Préserve les blocs de code tels quels (ne modifie rien entre ```...``` ou dans du code indenté).

SORTIE : Renvoie UNIQUEMENT le texte corrigé. Pas de commentaire, pas d'explication, pas de bavardage.
No yapping."""


@dataclass
class CorrectionResult:
    """Résultat de la correction LLM."""
    original_text: str
    corrected_text: str
    provider: str
    model: str
    corrections_count: int


class LLMCorrector:
    """
    Correcteur post-OCR utilisant un LLM.

    Providers supportés :
        - "openai"    : OpenAI API (GPT-4o-mini par défaut)
        - "anthropic" : Anthropic API (Claude 3.5 Sonnet par défaut)
        - "simulate"  : Mode simulation regex-based (pas d'API, fallback offline)

    Args:
        provider: Le provider LLM à utiliser.
        api_key: Clé API. Si None, cherche dans les variables d'environnement.
        model: Nom du modèle. Si None, utilise le modèle par défaut du provider.
        temperature: Température de génération (0.0 = déterministe).
    """

    # Modèles par défaut par provider
    DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-20241022",
    }

    def __init__(
        self,
        provider: Literal["openai", "anthropic", "simulate"] = "simulate",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.provider = provider
        self.temperature = temperature
        self.model = model or self.DEFAULT_MODELS.get(provider, "simulate")

        # Résolution de l'API key
        if provider == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                logger.warning(
                    "OPENAI_API_KEY non trouvée. Basculement en mode simulation."
                )
                self.provider = "simulate"
        elif provider == "anthropic":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                logger.warning(
                    "ANTHROPIC_API_KEY non trouvée. Basculement en mode simulation."
                )
                self.provider = "simulate"
        else:
            self.api_key = None

        logger.info(f"LLMCorrector initialisé — provider={self.provider}, model={self.model}")

    def correct(self, raw_text: str) -> CorrectionResult:
        """
        Corrige le texte brut OCR.

        Args:
            raw_text: Texte brut issu de l'OCR.

        Returns:
            CorrectionResult avec le texte corrigé.
        """
        if not raw_text.strip():
            return CorrectionResult(
                original_text=raw_text,
                corrected_text=raw_text,
                provider=self.provider,
                model=self.model,
                corrections_count=0,
            )

        logger.info(f"Correction LLM — {len(raw_text)} caractères via {self.provider}")

        if self.provider == "openai":
            corrected = self._call_openai(raw_text)
        elif self.provider == "anthropic":
            corrected = self._call_anthropic(raw_text)
        else:
            corrected = self._simulate_correction(raw_text)

        # Compter les corrections (diff simple)
        corrections = sum(
            1 for a, b in zip(raw_text.split(), corrected.split()) if a != b
        )

        return CorrectionResult(
            original_text=raw_text,
            corrected_text=corrected,
            provider=self.provider,
            model=self.model,
            corrections_count=corrections,
        )

    # ────────────────────────────────────────
    # Provider : OpenAI
    # ────────────────────────────────────────
    def _call_openai(self, text: str) -> str:
        """Appel à l'API OpenAI."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
            )
            corrected = response.choices[0].message.content.strip()
            logger.info(f"OpenAI — réponse reçue ({len(corrected)} chars)")
            return corrected

        except ImportError:
            logger.error("Package 'openai' non installé. pip install openai")
            return self._simulate_correction(text)
        except Exception as e:
            logger.error(f"Erreur OpenAI : {e}")
            return self._simulate_correction(text)

    # ────────────────────────────────────────
    # Provider : Anthropic
    # ────────────────────────────────────────
    def _call_anthropic(self, text: str) -> str:
        """Appel à l'API Anthropic."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": text},
                ],
            )
            corrected = response.content[0].text.strip()
            logger.info(f"Anthropic — réponse reçue ({len(corrected)} chars)")
            return corrected

        except ImportError:
            logger.error("Package 'anthropic' non installé. pip install anthropic")
            return self._simulate_correction(text)
        except Exception as e:
            logger.error(f"Erreur Anthropic : {e}")
            return self._simulate_correction(text)

    # ────────────────────────────────────────
    # Mode simulation (offline / fallback)
    # ────────────────────────────────────────
    def _simulate_correction(self, text: str) -> str:
        """
        Correction heuristique par regex — fallback sans API.

        Corrige les erreurs OCR les plus fréquentes sur le vocabulaire
        technique ML/DL. Utile pour les tests et démos sans clé API.
        """
        logger.info("Mode simulation — correction regex-based")

        # Dictionnaire de corrections OCR fréquentes (case-insensitive → casse correcte)
        corrections = {
            # ── Frameworks & Librairies ──
            r"\bPyToroh\b": "PyTorch",
            r"\bPytoroh\b": "PyTorch",
            r"\bPyTorc\b": "PyTorch",
            r"\bpytoroh\b": "pytorch",
            r"\bTensorFIow\b": "TensorFlow",
            r"\bTensorf1ow\b": "TensorFlow",
            r"\bTensorfiow\b": "TensorFlow",
            r"\btensorf1ow\b": "tensorflow",
            r"\bKéras\b": "Keras",
            r"\bScikit-Iearn\b": "Scikit-learn",
            r"\bscikit-1earn\b": "scikit-learn",
            r"\bNumPy\b": "NumPy",
            r"\bnumDy\b": "numpy",
            r"\bHuggingFace\b": "Hugging Face",

            # ── Vocabulaire ML/DL (français) ──
            r"\bréseau de neuronnes\b": "réseau de neurones",
            r"\bréseaux de neuronnes\b": "réseaux de neurones",
            r"\bneuronnes\b": "neurones",
            r"\bconvolutionne1\b": "convolutionnel",
            r"\bconvolutionnel1e\b": "convolutionnelle",
            r"\brétropropagatiom\b": "rétropropagation",
            r"\brétropropagatlon\b": "rétropropagation",
            r"\bapprenti[sz]sage\b": "apprentissage",
            r"\bapprentlssage\b": "apprentissage",
            r"\bsupervlsé\b": "supervisé",
            r"\bnon[ -]supervlsé\b": "non supervisé",
            r"\bclassificatiom\b": "classification",
            r"\brégressiom\b": "régression",
            r"\boptimisatiom\b": "optimisation",
            r"\brégularisatiom\b": "régularisation",
            r"\bnormalisatiom\b": "normalisation",

            # ── Vocabulaire ML/DL (anglais) ──
            r"\bgradlent\b": "gradient",
            r"\bgradiant\b": "gradient",
            r"\bdescente de gradiant\b": "descente de gradient",
            r"\bbackpropagatiom\b": "backpropagation",
            r"\boverfltting\b": "overfitting",
            r"\bunderfltting\b": "underfitting",
            r"\bbatch slze\b": "batch size",
            r"\blearn1ng rate\b": "learning rate",
            r"\blearning rat\b": "learning rate",
            r"\bcross-entrooy\b": "cross-entropy",
            r"\bsoftrnax\b": "softmax",
            r"\bsoftrmax\b": "softmax",
            r"\bfonctlon\b": "fonction",
            r"\bfonction d'activatiom\b": "fonction d'activation",
            r"\bperceptrom\b": "perceptron",
            r"\btransforner\b": "transformer",
            r"\battentiom\b": "attention",
            r"\bencoder\b": "encoder",
            r"\bdécoder\b": "decoder",
            r"\btokenisatiom\b": "tokenisation",
            r"\bembeddinq\b": "embedding",
            r"\bembeddlng\b": "embedding",

            # ── Sigles & acronymes ──
            r"\bCN[NM]\b": "CNN",
            r"\bRN[NM]\b": "RNN",
            r"\bLSTlvl\b": "LSTM",
            r"\bLSTM\b": "LSTM",
            r"\bGA[NM]\b": "GAN",
            r"\bVA[EF]\b": "VAE",
            r"\bNL[PQ]\b": "NLP",
            r"\bOC[RB]\b": "OCR",
        }

        corrected = text
        for pattern, replacement in corrections.items():
            corrected = re.sub(pattern, replacement, corrected)

        return corrected


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_text = """
Le reseau de neuronnes convolutionne1 (CNM) utilise PyToroh pour
l'apprentlssage supervlsé. La descente de gradiant et la
rétropropagatiom sont les bases de l'optimisatiom.

Le TensorFIow framework est une alternative à PyToroh.
La fonctlon d'activatiom softrnax est utilisée en classificatiom.
"""

    corrector = LLMCorrector(provider="simulate")
    result = corrector.correct(test_text)

    print("═" * 60)
    print("ORIGINAL :")
    print(result.original_text)
    print("═" * 60)
    print("CORRIGÉ :")
    print(result.corrected_text)
    print("═" * 60)
    print(f"Corrections : {result.corrections_count}")
    print(f"Provider : {result.provider} ({result.model})")
