"""
Détecteur de Régime de Marché — Hidden Markov Model (HMM) à 3 états.

Sources académiques et institutionnelles :
  - Hamilton (1989)         : "A New Approach to the Economic Analysis of
                               Nonstationary Time Series" — Econometrica
                               Papier fondateur du HMM en économie, cité 10 000+ fois
  - Ang & Bekaert (2002)    : "Regime Switches in Interest Rates"
                               J. of Business & Economic Statistics (Belgique/USA)
                               → classification des états HMM par volatilité
  - Man AHL (UK)            : publie sa méthodologie de régime pour commodités
                               Trend-following fund, $40B+ AUM
  - High-Flyer 幻方量化 (Chine): approche multi-initialisations pour éviter
                               les minima locaux du HMM — technique standard
                               dans les firmes quant chinoises
  - hmmlearn library        : implémentation scikit-learn compatible,
                               peer-reviewed, utilisée dans des dizaines de papiers

3 états détectés :
  RANGE  (0) — faible volatilité, marché en consolidation
  TREND  (1) — volatilité moyenne, tendance directionnelle
  CHAOS  (2) — forte volatilité, news macro / événement imprévu
               → NE PAS TRADER en régime CHAOS
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger

from hmmlearn import hmm
from src.config import HMM_N_STATES, HMM_LOOKBACK, REGIME_NAMES, MODELS_DIR
from src.database import load_gold_prices, save_regime, get_last_regime


class RegimeDetector:
    """
    Détecte le régime de marché actuel via un Gaussian HMM à 3 états.

    Architecture inspirée de Man AHL (UK) et High-Flyer (Chine) :
    - Features : rendements log, volatilité rolling, ratio de volume
    - Classification des états par volatilité (Ang & Bekaert 2002)
    - Multi-initialisations pour robustesse (approche chinoise)
    - Persistence du modèle via joblib
    """

    MODEL_PATH = MODELS_DIR / "hmm_model.pkl"

    def __init__(self):
        self.model: hmm.GaussianHMM | None = None
        self.is_trained: bool = False
        self._state_to_regime: dict[int, str] = {}

    # ─────────────────────────────────────────────
    # Préparation des features HMM
    # ─────────────────────────────────────────────

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        3 features pour le HMM — choisies selon la littérature académique :
          1. Rendements log (Hamilton 1989 — caractérise la tendance)
          2. Volatilité rolling 20 (Ang & Bekaert 2002 — sépare RANGE/CHAOS)
          3. Ratio de volume (Man AHL — confirme les cassures)
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        returns    = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        volatility = returns.rolling(20).std().fillna(returns.std())

        if "volume" in df.columns and df["volume"].sum() > 0:
            sma_vol    = df["volume"].rolling(20).mean().replace(0, np.nan)
            vol_ratio  = (df["volume"] / sma_vol).fillna(1.0)
        else:
            vol_ratio = pd.Series(1.0, index=df.index)

        X = np.column_stack([
            returns.values,
            volatility.values,
            vol_ratio.values,
        ])
        return np.nan_to_num(X, nan=0.0)

    # ─────────────────────────────────────────────
    # Entraînement
    # ─────────────────────────────────────────────

    def train(self, df: pd.DataFrame, n_restarts: int = 10) -> float:
        """
        Entraîne le HMM sur les données gold.

        n_restarts : nombre d'initialisations (technique High-Flyer Chine)
                     → garde le modèle avec le meilleur score de log-vraisemblance
                     → évite les minima locaux de l'algorithme EM

        Returns : score de log-vraisemblance du meilleur modèle
        """
        X = self.prepare_features(df)
        X = X[-HMM_LOOKBACK:]

        best_model = None
        best_score = -np.inf

        logger.info(f"  HMM : entraînement sur {len(X)} bougies ({n_restarts} initialisations)...")

        for i in range(n_restarts):
            try:
                candidate = hmm.GaussianHMM(
                    n_components=HMM_N_STATES,
                    covariance_type="full",
                    n_iter=200,
                    tol=1e-4,
                    random_state=i * 42,
                )
                candidate.fit(X)
                score = candidate.score(X)
                if score > best_score:
                    best_score = score
                    best_model = candidate
            except Exception:
                continue

        if best_model is None:
            logger.error("❌ HMM : aucun modèle convergé")
            return -np.inf

        self.model = best_model
        self.is_trained = True
        self._build_state_mapping(X)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "mapping": self._state_to_regime}, self.MODEL_PATH)
        logger.success(f"✅ HMM entraîné — log-vraisemblance : {best_score:.2f}")
        logger.info(f"  Mapping états → régimes : {self._state_to_regime}")
        return best_score

    # ─────────────────────────────────────────────
    # Prédiction
    # ─────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> tuple[str, int, float]:
        """
        Prédit le régime actuel.

        Returns :
            regime_name   : "TREND" | "RANGE" | "CHAOS"
            regime_id     : 0 | 1 | 2
            confidence    : probabilité du régime (0.0 – 1.0)
        """
        if not self.is_trained:
            self.load()

        X = self.prepare_features(df)
        states = self.model.predict(X)
        probs  = self.model.predict_proba(X)

        last_state = int(states[-1])
        last_conf  = float(probs[-1][last_state])
        regime_name = self._state_to_regime.get(last_state, "RANGE")

        return regime_name, last_state, last_conf

    def get_regime_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retourne l'historique des régimes sur tout le DataFrame."""
        if not self.is_trained:
            self.load()

        X = self.prepare_features(df)
        states = self.model.predict(X)
        probs  = self.model.predict_proba(X)

        result = pd.DataFrame({
            "time":       df.index,
            "state_id":   states,
            "regime":     [self._state_to_regime.get(s, "RANGE") for s in states],
            "confidence": [float(probs[i][states[i]]) for i in range(len(states))],
        })
        return result

    # ─────────────────────────────────────────────
    # Classification des états (Ang & Bekaert 2002)
    # ─────────────────────────────────────────────

    def _build_state_mapping(self, X: np.ndarray):
        """
        Mappe les états HMM (numérotation aléatoire) aux régimes nommés.

        Méthode Ang & Bekaert (2002) :
          → Trier les états par variance des rendements (covars_[:, 0, 0])
          → Plus faible variance = RANGE
          → Variance intermédiaire = TREND
          → Plus forte variance = CHAOS
        """
        variances = [self.model.covars_[i][0][0] for i in range(HMM_N_STATES)]
        sorted_states = np.argsort(variances)

        self._state_to_regime = {
            int(sorted_states[0]): "RANGE",
            int(sorted_states[1]): "TREND",
            int(sorted_states[2]): "CHAOS",
        }

        # Log des statistiques par état pour validation
        for state_id, name in self._state_to_regime.items():
            mask    = self.model.predict(X) == state_id
            n       = mask.sum()
            pct     = n / len(X) * 100
            avg_ret = X[mask, 0].mean() * 100 if n > 0 else 0
            vol     = np.sqrt(variances[state_id]) * 100
            logger.info(f"    État {state_id} ({name}) : {pct:.1f}% du temps, vol={vol:.3f}%, ret_moy={avg_ret:.4f}%")

    # ─────────────────────────────────────────────
    # Persistance du modèle
    # ─────────────────────────────────────────────

    def save(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "mapping": self._state_to_regime}, self.MODEL_PATH)
        logger.info(f"💾 Modèle HMM sauvegardé → {self.MODEL_PATH}")

    def load(self) -> bool:
        if not self.MODEL_PATH.exists():
            logger.warning("⚠️ Modèle HMM non trouvé — entraînement requis")
            return False
        data = joblib.load(self.MODEL_PATH)
        self.model            = data["model"]
        self._state_to_regime = data["mapping"]
        self.is_trained       = True
        logger.info(f"✅ Modèle HMM chargé depuis {self.MODEL_PATH}")
        return True


# ─────────────────────────────────────────────
# Fonction de haut niveau — utilisée par main.py
# ─────────────────────────────────────────────

def train_and_save_regime_model(timeframe: str = "1h") -> RegimeDetector:
    """Charge les données, entraîne le HMM, sauvegarde et retourne le détecteur."""
    detector = RegimeDetector()
    df = load_gold_prices(timeframe=timeframe, days=730)

    if df.empty:
        logger.error("❌ Pas de données gold en base — lance d'abord src.main")
        return detector

    score = detector.train(df)

    if detector.is_trained:
        regime_name, regime_id, confidence = detector.predict(df)
        save_regime(
            time=df.index[-1],
            regime_name=regime_name,
            regime_id=regime_id,
            confidence=confidence,
            timeframe=timeframe,
        )
        logger.success(f"✅ Régime actuel : {regime_name} (confiance {confidence:.1%})")

    return detector


def get_current_regime(timeframe: str = "1h") -> tuple[str, float]:
    """
    Retourne le régime actuel depuis la base ou le recalcule si absent.
    Interface simple pour les autres modules.
    """
    detector = RegimeDetector()
    if not detector.load():
        logger.info("  Entraînement HMM initial...")
        detector = train_and_save_regime_model(timeframe)

    df = load_gold_prices(timeframe=timeframe, days=30)
    if df.empty or not detector.is_trained:
        return "RANGE", 0.5

    regime_name, _, confidence = detector.predict(df)
    return regime_name, confidence
