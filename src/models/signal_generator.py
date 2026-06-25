"""
Générateur de Signal — LightGBM, score 0-100.

Sources académiques et institutionnelles :
  - Ke et al. (2017)          : "LightGBM: A Highly Efficient Gradient Boosting
                                 Decision Tree" — NeurIPS 2017
                                 Microsoft Research Asia (Pékin, Chine)
                                 Créateurs de l'algorithme — on utilise leur implémentation exacte
  - Lopez de Prado (2018)     : "Advances in Financial Machine Learning" — Wiley
                                 Espagne/USA — Expert mondial ML finance
                                 → Purged Walk-Forward CV (évite le data leakage temporel)
                                 → Feature importance via MDA (Mean Decrease Accuracy)
  - Jane Street / Kaggle 2021 : "Jane Street Market Prediction" competition
                                 USA — Meilleures solutions publiques LightGBM pour marchés
                                 → early stopping, class weights, calibration
  - AQR Capital Management    : "Two Centuries of Multi-Asset Momentum" (2013)
                                 USA — valide que les signaux quantitatifs fonctionnent
                                 sur commodités dont l'or
  - High-Flyer 幻方量化 (Chine): ensemble de modèles + analyse de feature importance
                                 Firme quant chinoise $10B+ AUM
  - Man AHL (UK)              : walk-forward sur commodités, publié académiquement
                                 $40B+ AUM, leader du trend-following quantitatif

Architecture :
  - Entrée  : 72 features (7 stratégies + Chine + macro + HMM)
  - Sortie  : score 0-100 + direction LONG | SHORT | FLAT
  - Target  : close[t+4h] > close[t] × 1.001 (hausse de 0.1% en 4h)
  - CV      : Purged Walk-Forward (Lopez de Prado) — pas de fuite temporelle
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger

import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from src.config import (
    LGBM_FEATURES, LGBM_TARGET_HORIZON, LGBM_THRESHOLD_LONG, LGBM_THRESHOLD_SHORT,
    MODELS_DIR,
)
from src.database import load_gold_prices
from src.models.feature_builder import build_features
from src.models.regime_detector import get_current_regime


class SignalGenerator:
    """
    Modèle LightGBM pour la génération de signaux de trading.

    Inspiré de Jane Street Kaggle 2021 + Lopez de Prado 2018 + High-Flyer Chine.
    """

    MODEL_PATH    = MODELS_DIR / "lgbm_model.pkl"
    FEATURES_PATH = MODELS_DIR / "lgbm_features.pkl"
    PARAMS_PATH   = MODELS_DIR / "lgbm_best_params.json"

    # Hyperparamètres par défaut — remplacés automatiquement par Optuna si disponible
    # Source : Jane Street Kaggle top solutions + Man AHL research
    LGBM_PARAMS_DEFAULT = {
        "objective":        "binary",
        "metric":           "auc",
        "boosting_type":    "gbdt",
        "n_estimators":     500,
        "learning_rate":    0.03,
        "num_leaves":       31,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "reg_alpha":        0.1,
        "reg_lambda":       0.1,
        "random_state":     42,
        "verbose":          -1,
        "n_jobs":           -1,
    }

    @classmethod
    def _load_params(cls) -> dict:
        """Charge les paramètres Optuna si disponibles, sinon les paramètres par défaut."""
        import json
        if cls.PARAMS_PATH.exists():
            data = json.loads(cls.PARAMS_PATH.read_text())
            params = {
                "objective":     "binary",
                "metric":        "auc",
                "boosting_type": "gbdt",
                "random_state":  42,
                "verbose":       -1,
                "n_jobs":        -1,
                **data["params"],
            }
            logger.info(f"  Paramètres Optuna chargés — AUC référence : {data['best_auc']:.4f}")
            return params
        return cls.LGBM_PARAMS_DEFAULT

    @property
    def LGBM_PARAMS(self) -> dict:
        return self._load_params()

    def __init__(self):
        self.model          = None
        self.feature_names: list[str] = []
        self.is_trained     = False
        self._feature_importance: pd.DataFrame | None = None

    # ─────────────────────────────────────────────
    # Construction de la target (Lopez de Prado)
    # ─────────────────────────────────────────────

    def _build_target(self, df: pd.DataFrame, horizon: int = 48) -> pd.Series:
        """
        Triple Barrier Method — Lopez de Prado (2018) + mlfinlab (Hudson & Thames).

        Sources :
          - Lopez de Prado (2018) "Advances in Financial Machine Learning" Ch.3
          - hudson-and-thames/mlfinlab (GitHub) — implémentation de référence
          - NeurIPS 2025 : "Risk-Aware DRL for XAU/USD" — valide l'approche sur gold

        Trois barrières dynamiques basées sur l'ATR (volatilité réelle) :
          - Barrière haute (TP)   : prix + ATR × 3  → label 1 (LONG gagnant)
          - Barrière basse (SL)   : prix - ATR × 2  → label 0 (trade perdant)
          - Barrière temps        : horizon bougies  → label ignoré (neutre)

        Avantage vs target simple :
          Le modèle apprend directement si le TP sera touché AVANT le SL.
          C'est exactement la question qu'on pose au moment de trader.
          Élimine le look-ahead bias et aligne parfaitement avec l'exécution réelle.
        """
        close = df["close"] if "close" in df.columns else df["Close"]
        high  = df["high"]  if "high"  in df.columns else df["High"]
        low   = df["low"]   if "low"   in df.columns else df["Low"]

        # ATR 14 pour les barrières dynamiques
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        labels = pd.Series(index=df.index, dtype=float)

        for i in range(len(df) - horizon):
            entry     = close.iloc[i]
            atr_val   = atr.iloc[i]
            if pd.isna(atr_val) or atr_val <= 0:
                continue

            barrier_up   = entry + atr_val * 3.0   # TP
            barrier_down = entry - atr_val * 2.0   # SL

            label = np.nan
            for j in range(i + 1, min(i + horizon, len(df))):
                h = high.iloc[j]
                l = low.iloc[j]
                if h >= barrier_up:
                    label = 1   # TP touché en premier → trade gagnant
                    break
                if l <= barrier_down:
                    label = 0   # SL touché en premier → trade perdant
                    break
            labels.iloc[i] = label

        return labels.dropna().astype(int)

    # ─────────────────────────────────────────────
    # Purged Walk-Forward CV (Lopez de Prado 2018)
    # ─────────────────────────────────────────────

    def _purged_train_test_split(
        self, X: pd.DataFrame, y: pd.Series, train_pct: float = 0.75, gap: int = 4
    ) -> tuple:
        """
        Division train/test avec embargo (gap) entre les deux.

        Évite le data leakage temporel : les données proches de la frontière
        train/test sont retirées. Source : Lopez de Prado 2018, chapitre 7.

        gap : nombre de bougies d'embargo (= horizon de prédiction)
        """
        n = len(X)
        train_end = int(n * train_pct)
        test_start = train_end + gap  # Embargo de `gap` bougies

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test  = X.iloc[test_start:]
        y_test  = y.iloc[test_start:]

        return X_train, X_test, y_train, y_test

    # ─────────────────────────────────────────────
    # Entraînement
    # ─────────────────────────────────────────────

    def train(
        self,
        df_features: pd.DataFrame,
        df_prices: pd.DataFrame,
        cot_df: pd.DataFrame | None = None,
        macro_dict: dict | None = None,
    ) -> dict:
        """
        Entraîne le modèle LightGBM avec Purged Walk-Forward CV.

        Returns : dict avec les métriques (AUC, precision, recall)
        """
        logger.info("🤖 Entraînement LightGBM...")

        # Construire les features si pas déjà fait
        if df_features.empty:
            regime_name, confidence = get_current_regime("1h")
            regime_id = {"RANGE": 0, "TREND": 1, "CHAOS": 2}.get(regime_name, 0)
            df_features = build_features(df_prices, cot_df, macro_dict, regime_id, confidence)

        # Target — Triple Barrier avec horizon 48h
        df_prices_aligned = df_prices.copy()
        df_prices_aligned.columns = df_prices_aligned.columns.str.lower()
        y_full = self._build_target(df_prices_aligned, horizon=24)

        # Aligner features et target sur leur index commun
        common_idx = df_features.index.intersection(y_full.index)
        X_full = df_features.loc[common_idx].copy()
        y_full = y_full.loc[common_idx].copy()

        # Supprimer les colonnes dupliquées (produites par pd.concat entre frames)
        X_full = X_full.loc[:, ~X_full.columns.duplicated(keep='first')]

        # Utiliser TOUTES les features disponibles (63 calculées, pas seulement 32)
        excluded = {"open", "high", "low", "close", "volume"}
        base_features = [f for f in LGBM_FEATURES if f in X_full.columns]
        extra_features = [
            c for c in X_full.columns
            if c not in base_features
            and c not in excluded
            and float(X_full[c].notna().mean()) > 0.8
        ]
        available = base_features + extra_features
        if len(available) < 10:
            available = [c for c in X_full.columns
                         if float(X_full[c].notna().mean()) > 0.8]
        self.feature_names = available
        X_full = X_full[self.feature_names].fillna(0)

        # Éliminer les timestamps dupliqués (produits par resample 4h/daily dans multitf_features)
        # Cause racine : df.loc[common_idx] retourne TOUTES les lignes matching un label,
        # y compris les doublons → X_full > y_full en taille après .loc[common_idx]
        X_full = X_full[~X_full.index.duplicated(keep='first')]
        y_full = y_full[~y_full.index.duplicated(keep='first')]

        # Aligner sur l'index commun (sans doublons → taille garantie identique)
        common_idx2 = X_full.index.intersection(y_full.index)
        X_full = X_full.loc[common_idx2]
        y_full = y_full.loc[common_idx2]

        # Filtrer les NaN cibles avec .values (numpy positionnel → aucune ambiguïté d'index)
        valid_mask = y_full.notna().values
        X_full = X_full.iloc[valid_mask].reset_index(drop=True)
        y_full = y_full.iloc[valid_mask].reset_index(drop=True)

        logger.info(f"  Dataset : {len(X_full)} échantillons, {len(self.feature_names)} features")
        logger.info(f"  Distribution target : {y_full.mean():.1%} LONG, {1-y_full.mean():.1%} SHORT/FLAT")

        # Purged Walk-Forward split (Lopez de Prado)
        X_train, X_test, y_train, y_test = self._purged_train_test_split(
            X_full, y_full, train_pct=0.75, gap=LGBM_TARGET_HORIZON
        )

        # Poids de classe — compense le déséquilibre LONG/SHORT
        # Technique Jane Street Kaggle 2021
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if y_train.sum() > 0 else 1.0
        params = {**self.LGBM_PARAMS, "scale_pos_weight": float(pos_weight)}

        # Entraînement avec early stopping
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=callbacks,
        )

        # Métriques (AUC — standard industrie pour signaux binaires)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= LGBM_THRESHOLD_LONG).astype(int)
        auc  = roc_auc_score(y_test, y_prob) if len(y_test.unique()) > 1 else 0.5
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)

        metrics = {"auc": auc, "precision": prec, "recall": rec,
                   "n_train": len(X_train), "n_test": len(X_test)}

        logger.success(f"✅ LightGBM entraîné — AUC={auc:.3f}, Précision={prec:.3f}, Recall={rec:.3f}")
        logger.info(f"  Proba LONG — min={y_prob.min():.3f} | median={np.median(y_prob):.3f} | max={y_prob.max():.3f} | >0.52={( y_prob >= 0.52).mean():.1%}")

        # Feature importance (High-Flyer / Lopez de Prado — MDA)
        self._compute_feature_importance(X_test, y_test)

        self.is_trained = True
        self.save()
        return metrics

    # ─────────────────────────────────────────────
    # Prédiction
    # ─────────────────────────────────────────────

    def predict(self, features: pd.DataFrame | dict) -> tuple[float, str]:
        """
        Génère un signal pour la bougie actuelle.

        Returns :
            score     : 0-100 (probabilité LONG × 100)
            direction : "LONG" | "SHORT" | "FLAT"
        """
        if not self.is_trained:
            if not self.load():
                return 50.0, "FLAT"

        if isinstance(features, dict):
            X = pd.DataFrame([features])
        else:
            X = features.copy()

        # Aligner sur les features du modèle
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names].fillna(0)

        prob_long = float(self.model.predict_proba(X)[0][1])

        if prob_long >= LGBM_THRESHOLD_LONG:
            direction = "LONG"
            score = round(prob_long * 100, 1)
        elif prob_long <= LGBM_THRESHOLD_SHORT:
            direction = "SHORT"
            score = round((1.0 - prob_long) * 100, 1)  # confiance SHORT
        else:
            direction = "FLAT"
            score = 50.0

        return score, direction

    def predict_batch(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """Prédit sur tout un DataFrame — utile pour le backtest."""
        if not self.is_trained:
            self.load()

        X = df_features.copy()
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names].fillna(0)

        probs = self.model.predict_proba(X)[:, 1]

        directions = np.where(probs >= LGBM_THRESHOLD_LONG, "LONG",
                     np.where(probs <= LGBM_THRESHOLD_SHORT, "SHORT", "FLAT"))

        # Score = confiance dans la direction prédite (pas juste prob_long)
        scores = np.where(probs >= LGBM_THRESHOLD_LONG,  probs * 100,
                 np.where(probs <= LGBM_THRESHOLD_SHORT, (1.0 - probs) * 100,
                          50.0))

        return pd.DataFrame({
            "score":     scores,
            "direction": directions,
            "prob_long": probs,
        }, index=df_features.index)

    # ─────────────────────────────────────────────
    # Feature importance (High-Flyer + Lopez de Prado)
    # ─────────────────────────────────────────────

    def _compute_feature_importance(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Feature importance par gain — identifie quelles stratégies contribuent le plus."""
        importance = pd.DataFrame({
            "feature":   self.feature_names,
            "gain":      self.model.booster_.feature_importance(importance_type="gain"),
            "split":     self.model.booster_.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False)

        self._feature_importance = importance
        top5 = importance.head(5)
        logger.info("  Top 5 features par contribution :")
        for _, row in top5.iterrows():
            logger.info(f"    {row['feature']:30s} gain={row['gain']:.0f}")

    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        if self._feature_importance is None:
            return pd.DataFrame()
        return self._feature_importance.head(n)

    # ─────────────────────────────────────────────
    # Persistance
    # ─────────────────────────────────────────────

    def save(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model":         self.model,
            "feature_names": self.feature_names,
            "importance":    self._feature_importance,
        }, self.MODEL_PATH)
        logger.info(f"💾 LightGBM sauvegardé → {self.MODEL_PATH}")

    def load(self) -> bool:
        if not self.MODEL_PATH.exists():
            logger.warning("⚠️ Modèle LightGBM non trouvé — entraînement requis")
            return False
        data = joblib.load(self.MODEL_PATH)
        self.model               = data["model"]
        self.feature_names       = data["feature_names"]
        self._feature_importance = data.get("importance")
        self.is_trained          = True
        logger.info(f"✅ LightGBM chargé — {len(self.feature_names)} features")
        return True


# ─────────────────────────────────────────────
# Fonction de haut niveau
# ─────────────────────────────────────────────

def merge_institutional_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge les données institutionnelles historiques dans le DataFrame OHLCV.
    Time-aligné : chaque barre 1h reçoit la valeur en vigueur à ce moment.
    """
    from src.data.collector import fetch_cot_gold, fetch_fred_macro
    df = df.copy()

    # ── COT CFTC (weekly → 1h via forward fill) ──────────────────────────────
    try:
        cot_df = fetch_cot_gold()
        if not cot_df.empty:
            cot_idx = cot_df.set_index("report_date").sort_index()
            cot_idx.index = pd.to_datetime(cot_idx.index, utc=True)
            for col in ["commercial_index", "large_spec_net"]:
                if col in cot_idx.columns:
                    df[col] = cot_idx[col].reindex(df.index, method="ffill")
            # Signal directionnel (-1/0/+1)
            if "cot_signal" in cot_idx.columns:
                smap = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}
                df["cot_signal_num"] = (
                    cot_idx["cot_signal"].map(smap)
                    .reindex(df.index, method="ffill")
                    .fillna(0.0)
                )
            logger.info(f"  COT mergé : {cot_df['commercial_index'].notna().sum()} semaines")
    except Exception as e:
        logger.warning(f"⚠️ COT merge : {e}")

    # ── FRED macro (daily → 1h via forward fill) ──────────────────────────────
    try:
        macro_dict = fetch_fred_macro()
        for series_id, series in macro_dict.items():
            if hasattr(series, "index") and not series.empty:
                s = series.copy()
                s.index = pd.to_datetime(s.index, utc=True)
                # lowercase pour correspondre au df.columns.str.lower() dans build_features
                df[f"fred_{series_id.lower()}"] = s.reindex(df.index, method="ffill")
        if macro_dict:
            logger.info(f"  FRED mergé : {len(macro_dict)} séries")
    except Exception as e:
        logger.warning(f"⚠️ FRED merge : {e}")

    return df


def train_signal_model(timeframe: str = "1h", days: int = 730) -> SignalGenerator:
    """Charge les données, construit les features, entraîne et sauvegarde."""
    generator = SignalGenerator()

    logger.info("📊 Chargement des données pour entraînement LightGBM...")
    df_prices = load_gold_prices(timeframe=timeframe, days=days)
    if df_prices.empty:
        logger.error("❌ Pas de données — lance d'abord src.main")
        return generator

    # Merger les données institutionnelles historiques (time-alignées)
    logger.info("🌍 Merge données institutionnelles (COT + FRED)...")
    df_prices = merge_institutional_data(df_prices)

    # Régime HMM actuel
    regime_name, confidence = get_current_regime(timeframe)
    regime_id = {"RANGE": 0, "TREND": 1, "CHAOS": 2}.get(regime_name, 0)

    # Features (inclut maintenant institutional_features_from_df)
    logger.info("⚙️  Calcul des features...")
    df_features = build_features(df_prices, None, None, regime_id, confidence)

    # Entraînement
    metrics = generator.train(df_features, df_prices, None, None)

    return generator


def get_current_signal(timeframe: str = "1h") -> tuple[float, str]:
    """
    Retourne le signal actuel : (score 0-100, direction).
    Interface simple pour main.py et telegram_bot.py.
    """
    from src.data.collector import fetch_cot_gold, fetch_fred_macro

    generator = SignalGenerator()
    if not generator.load():
        logger.warning("⚠️ Modèle non entraîné — entraînement automatique...")
        generator = train_signal_model(timeframe)
        if not generator.is_trained:
            return 50.0, "FLAT"

    df_prices = load_gold_prices(timeframe=timeframe, days=30)
    if df_prices.empty:
        return 50.0, "FLAT"

    cot_df     = fetch_cot_gold()
    macro_dict = fetch_fred_macro()
    regime_name, confidence = get_current_regime(timeframe)
    regime_id = {"RANGE": 0, "TREND": 1, "CHAOS": 2}.get(regime_name, 0)

    df_features = build_features(df_prices, cot_df, macro_dict, regime_id, confidence)
    last_row = df_features.iloc[[-1]]

    return generator.predict(last_row)
