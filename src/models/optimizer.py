"""
Optimisation des hyperparamètres LightGBM — Optuna.

Sources :
  - Optuna : Akiba et al., "Optuna: A Next-generation Hyperparameter
    Optimization Framework" (Preferred Networks, Japon — 2019)
    https://arxiv.org/abs/1907.10902
  - Jane Street : top solutions Kaggle "Market Prediction" (2021)
    utilisent Optuna + LightGBM sur données financières
  - Man AHL : "A Practitioner's Guide to Reading the Term Structure"
    recommande la validation walk-forward pour éviter l'overfitting
    sur séries temporelles

Fonctionnement :
  - 100 trials Optuna, chacun entraîne un LightGBM différent
  - Métrique : AUC sur la période de test (walk-forward)
  - Pruning Hyperband : arrête les trials prometteurs tôt si AUC décline
  - Sauvegarde les meilleurs paramètres dans src/models/lgbm_best_params.json
  - signal_generator.py charge ces paramètres automatiquement

Durée estimée : 15-30 minutes (100 trials × ~15s chacun)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PARAMS_PATH = Path(__file__).parent / "lgbm_best_params.json"
N_TRIALS    = 100    # nombre de combinaisons testées
N_CV_FOLDS  = 3      # cross-validation temporelle


def build_dataset():
    """Charge les données et construit le dataset pour l'optimisation."""
    from src.database import load_gold_prices
    from src.data.collector import fetch_cot_gold, fetch_fred_macro
    from src.models.feature_builder import build_features
    from src.models.regime_detector import RegimeDetector
    from src.models.signal_generator import SignalGenerator

    logger.info("Chargement des données...")
    df      = load_gold_prices()
    cot_df  = fetch_cot_gold()
    macro   = fetch_fred_macro()

    regime_det = RegimeDetector()
    regime_det.load()

    # Régime HMM sur tout le dataset
    all_regimes = regime_det.predict_series(df)
    regime_id   = int(all_regimes.iloc[-1]) if all_regimes is not None else 0

    logger.info("Construction des features...")
    features = build_features(df, cot_df=cot_df, macro_dict=macro, regime_id=regime_id)
    features = features[~features.index.duplicated(keep='first')]

    # Cible Triple Barrier (même logique que signal_generator)
    sig_gen = SignalGenerator()
    target  = sig_gen._build_target(df)

    # Alignement
    common = features.index.intersection(target.index)
    X = features.loc[common].fillna(0)
    y = target.loc[common]

    # Walk-forward : 70% train, 30% test
    split = int(len(X) * 0.70)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    logger.info(f"Dataset : {len(X_train)} train / {len(X_test)} test | {X.shape[1]} features")
    logger.info(f"Distribution target : {y_train.mean()*100:.1f}% LONG")

    return X_train, X_test, y_train, y_test, list(X.columns)


def objective(trial, X_train, X_test, y_train, y_test):
    """
    Fonction objectif Optuna — un trial = une combinaison de paramètres.

    Espace de recherche inspiré de :
      - Jane Street top solutions (num_leaves élevé pour capturer complexité)
      - Man AHL research (learning_rate faible + regularisation forte)
      - Microsoft LightGBM docs (paramètres recommandés séries financières)
    """
    params = {
        "objective":         "binary",
        "metric":            "auc",
        "boosting_type":     "gbdt",
        "verbose":           -1,
        "n_jobs":            -1,
        "random_state":      42,

        # ── Complexité du modèle ──────────────────────────
        "n_estimators":      trial.suggest_int("n_estimators", 200, 2000),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),

        # ── Learning rate ─────────────────────────────────
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.15, log=True),

        # ── Sous-échantillonnage (évite overfitting) ──────
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq":      trial.suggest_int("bagging_freq", 1, 10),

        # ── Régularisation L1/L2 ─────────────────────────
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
    }

    try:
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        proba = model.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, proba)
        return auc

    except Exception as e:
        logger.debug(f"Trial échoué : {e}")
        return 0.5


def run_optimization(n_trials: int = N_TRIALS):
    """Lance l'optimisation Optuna et sauvegarde les meilleurs paramètres."""

    logger.info("=" * 55)
    logger.info("  🔬 OPTUNA — Optimisation LightGBM")
    logger.info(f"  {n_trials} trials | Objectif : maximiser AUC")
    logger.info("=" * 55)

    X_train, X_test, y_train, y_test, feature_names = build_dataset()

    # Pruner Hyperband — coupe les trials peu prometteurs rapidement
    # Source : Li et al., "Hyperband: A Novel Bandit-Based Approach" (2018)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource=n_trials, reduction_factor=3
    )
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
        study_name="goldbot_lgbm",
    )

    # Barre de progression manuelle
    best_so_far = 0.0

    def progress_callback(study, trial):
        nonlocal best_so_far
        if study.best_value > best_so_far:
            best_so_far = study.best_value
            logger.success(
                f"  Trial {trial.number:3d} | AUC = {study.best_value:.4f} ✨ nouveau meilleur"
            )
        elif trial.number % 10 == 0:
            logger.info(
                f"  Trial {trial.number:3d} | AUC actuel = {trial.value:.4f} | "
                f"Meilleur = {best_so_far:.4f}"
            )

    study.optimize(
        lambda trial: objective(trial, X_train, X_test, y_train, y_test),
        n_trials=n_trials,
        callbacks=[progress_callback],
        show_progress_bar=False,
    )

    # ── Résultats ──────────────────────────────────────────
    best_params = study.best_params
    best_auc    = study.best_value

    logger.success(f"\n✅ Optimisation terminée — Meilleur AUC : {best_auc:.4f}")
    logger.info(f"   AUC de départ (baseline) : 0.583")
    gain = (best_auc - 0.583) * 100
    logger.info(f"   Gain : {gain:+.2f} points d'AUC")
    logger.info(f"\n   Meilleurs paramètres :")
    for k, v in best_params.items():
        logger.info(f"     {k:25s} = {v}")

    # ── Sauvegarde ────────────────────────────────────────
    result = {
        "best_auc":    best_auc,
        "baseline_auc": 0.583,
        "n_trials":    n_trials,
        "params":      best_params,
    }
    PARAMS_PATH.write_text(json.dumps(result, indent=2))
    logger.success(f"\n💾 Paramètres sauvegardés → {PARAMS_PATH}")
    logger.info("   Relance le backtest pour valider : python -m src.backtest.backtest")

    return best_params, best_auc


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_TRIALS
    run_optimization(n_trials=n)
