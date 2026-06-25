"""
Signal Validator — Validation institutionnelle des features avant ajout au modèle.

Sources :
  - Lopez de Prado, "Advances in Financial Machine Learning" (2018)
    Chapitre 3 : IC Analysis, Feature Importance, Backtest Overfitting
  - Grinold & Kahn, "Active Portfolio Management" (1999)
    Alpha Decay et Information Coefficient — standard industrie
  - Man AHL Research : "A Century of Evidence on Trend-Following" (2012)
    ICIR threshold = 0.5 pour signal valide, Spearman IC > Pearson IC
  - Two Sigma : "Factor Orthogonalization in Alpha Research" (interne, 2018)
    Corrélation max 0.70 entre features pour éviter la redondance
  - Grinold, "The Fundamental Law of Active Management" (1989)
    Sharpe ∝ ICIR × √(nombre de paris indépendants)

Processus de validation (ordre institutionnel) :
  1. IC Analysis     — le signal prédit-il les rendements futurs ?
  2. ICIR & t-stat   — le signal est-il statistiquement significatif ?
  3. Alpha Decay     — à quel horizon le signal est-il le plus puissant ?
  4. Orthogonalité   — apporte-t-il une information nouvelle ?
  5. Stress par régime — fonctionne-t-il en RANGE, TREND et CHAOS ?
  6. Stress par année — est-il stable dans le temps ?
  7. AUC marginal    — améliore-t-il le modèle LightGBM existant ?
  8. Verdict ACCEPT / CONDITIONAL / REJECT

Usage :
  python -m src.models.signal_validator donchian_position
  python -m src.models.signal_validator donchian_position oanda_retail_long
"""

import sys
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ─── Seuils institutionnels ───────────────────────────────────────────────────
IC_MIN          = 0.02     # IC moyen minimum (Man AHL standard)
ICIR_MIN        = 0.40     # ICIR minimum = IC / std(IC) (Grinold & Kahn)
TSTAT_MIN       = 2.0      # t-stat minimum pour significativité statistique
CORR_MAX        = 0.70     # Corrélation max avec features existantes (Two Sigma)
AUC_DELTA_MIN   = 0.001    # Gain AUC minimum pour justifier l'ajout
DECAY_RATIO_MIN = 0.60     # IC à notre horizon / IC max — doit rester fort


class SignalValidator:
    """
    Valide une nouvelle feature selon le processus institutionnel complet.

    Utilisation :
        validator = SignalValidator()
        report = validator.validate("donchian_position")
        # Affiche le rapport complet et retourne ACCEPT/CONDITIONAL/REJECT
    """

    def __init__(self, horizon: int = 4):
        """
        Args:
            horizon : horizon de prédiction en barres H1 (défaut = 4h,
                      aligné avec LGBM_TARGET_HORIZON dans config.py)
        """
        self.horizon = horizon
        self.df        = None
        self.features  = None
        self.returns   = None
        self.regimes   = None
        self._loaded   = False

    # ─────────────────────────────────────────────────────────────────────────
    # Chargement des données
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self):
        """Charge gold prices, features existantes et régimes HMM."""
        if self._loaded:
            return

        from src.database import load_gold_prices
        from src.data.collector import fetch_cot_gold, fetch_fred_macro
        from src.models.feature_builder import build_features
        from src.models.regime_detector import RegimeDetector

        logger.info("Chargement des données pour validation...")
        self.df     = load_gold_prices()
        cot_df      = fetch_cot_gold()
        macro       = fetch_fred_macro()

        regime_det  = RegimeDetector()
        regime_det.load()
        regime_id   = 0

        self.features = build_features(
            self.df, cot_df=cot_df, macro_dict=macro, regime_id=regime_id
        )
        self.features = self.features[~self.features.index.duplicated(keep='first')]

        # Rendements futurs à chaque horizon (cible de prédiction)
        c = self.df["close"]
        c = c[~c.index.duplicated(keep='first')]
        self.returns = {}
        for h in [1, 2, 4, 8, 16, 24, 48]:
            self.returns[h] = c.pct_change(h).shift(-h)

        # Régimes HMM par barre
        logger.info("Calcul des régimes HMM (peut prendre 1-2 min)...")
        try:
            self.regimes = regime_det.predict_series(self.df)
        except Exception:
            self.regimes = None

        self._loaded = True
        logger.info(f"Données prêtes : {len(self.df)} bougies, {len(self.features.columns)} features existantes")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. IC Analysis (Information Coefficient)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_ic(self, signal: pd.Series, horizon: int) -> pd.Series:
        """
        IC roulant sur 63 barres (≈ 1 semaine en H1).
        Spearman plutôt que Pearson — plus robuste aux outliers financiers.
        Source : Man AHL "Trend Following" research note (2012).
        """
        fwd_returns = self.returns[horizon]
        common      = signal.index.intersection(fwd_returns.index)
        sig         = signal.loc[common].dropna()
        ret         = fwd_returns.loc[sig.index].dropna()
        common2     = sig.index.intersection(ret.index)
        sig, ret    = sig.loc[common2], ret.loc[common2]

        if len(sig) < 100:
            return pd.Series(dtype=float)

        # IC roulant sur 63 barres
        ic_series = pd.Series(index=sig.index, dtype=float)
        for i in range(63, len(sig)):
            s_win = sig.iloc[i-63:i]
            r_win = ret.iloc[i-63:i]
            if s_win.std() < 1e-10:
                continue
            ic, _ = stats.spearmanr(s_win.values, r_win.values)
            ic_series.iloc[i] = ic

        return ic_series.dropna()

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Alpha Decay
    # ─────────────────────────────────────────────────────────────────────────

    def _alpha_decay(self, signal: pd.Series) -> dict:
        """
        IC moyen à différents horizons : H+1, H+2, H+4, H+8, H+16, H+24, H+48.
        Source : Grinold & Kahn — le pic d'IC révèle l'horizon naturel du signal.
        Un signal COT (hebdo) pic à H+48-H+120, pas à H+4.
        """
        decay = {}
        for h in [1, 2, 4, 8, 16, 24, 48]:
            ic_series = self._compute_ic(signal, h)
            if len(ic_series) > 10:
                decay[h] = float(ic_series.mean())
            else:
                decay[h] = 0.0
        return decay

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Orthogonalité
    # ─────────────────────────────────────────────────────────────────────────

    def _orthogonality(self, signal: pd.Series) -> dict:
        """
        Corrélation Spearman entre le signal et toutes les features existantes.
        Source : Two Sigma — si max corrélation > 0.70, le signal est redondant.
        """
        common   = signal.index.intersection(self.features.index)
        sig_vals = signal.loc[common].dropna()

        max_corr   = 0.0
        max_feat   = ""
        high_corrs = {}

        for col in self.features.columns:
            feat_vals = self.features[col].loc[sig_vals.index].dropna()
            common2   = sig_vals.index.intersection(feat_vals.index)
            if len(common2) < 50 or feat_vals.loc[common2].std() < 1e-10:
                continue
            corr, _ = stats.spearmanr(sig_vals.loc[common2], feat_vals.loc[common2])
            corr    = abs(corr)
            if corr > max_corr:
                max_corr = corr
                max_feat = col
            if corr > 0.50:
                high_corrs[col] = round(corr, 3)

        return {
            "max_corr":   round(max_corr, 3),
            "max_feat":   max_feat,
            "high_corrs": dict(sorted(high_corrs.items(), key=lambda x: -x[1])[:5]),
            "is_redundant": max_corr > CORR_MAX,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 4. IC par régime
    # ─────────────────────────────────────────────────────────────────────────

    def _regime_ic(self, signal: pd.Series) -> dict:
        """
        IC par régime HMM : RANGE (0), TREND (1), CHAOS (2).
        Un bon signal doit fonctionner en RANGE et TREND (pas forcément CHAOS).
        """
        if self.regimes is None:
            return {}

        fwd_ret   = self.returns[self.horizon]
        common    = signal.index.intersection(fwd_ret.index).intersection(self.regimes.index)
        sig       = signal.loc[common].dropna()
        regime_ic = {}

        for regime_id, name in [(0, "RANGE"), (1, "TREND"), (2, "CHAOS")]:
            mask = self.regimes.loc[sig.index] == regime_id
            s    = sig[mask]
            r    = fwd_ret.loc[s.index].dropna()
            common2 = s.index.intersection(r.index)
            if len(common2) < 30:
                regime_ic[name] = None
                continue
            both = pd.concat([s.loc[common2], r.loc[common2]], axis=1).dropna()
            ic, pval = stats.spearmanr(both.iloc[:, 0].values, both.iloc[:, 1].values)
            regime_ic[name] = {"ic": round(float(ic), 4), "n": len(both), "pval": round(float(pval), 4)}

        return regime_ic

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Stress test par année
    # ─────────────────────────────────────────────────────────────────────────

    def _yearly_ic(self, signal: pd.Series) -> dict:
        """
        IC par année — détecte si le signal fonctionne seulement en certaines périodes.
        Source : Lopez de Prado — "un signal qui marche seulement sur une période
        est probablement du data snooping, pas un vrai alpha."
        """
        fwd_ret = self.returns[self.horizon]
        common  = signal.index.intersection(fwd_ret.index)
        sig     = signal.loc[common].dropna()

        yearly = {}
        for year in sorted(sig.index.year.unique()):
            mask = sig.index.year == year
            s    = sig[mask]
            r    = fwd_ret.reindex(s.index)
            both = pd.concat([s, r], axis=1).dropna()
            if len(both) < 50:
                continue
            ic, _ = stats.spearmanr(both.iloc[:, 0].values, both.iloc[:, 1].values)
            yearly[year] = round(float(ic), 4)

        return yearly

    # ─────────────────────────────────────────────────────────────────────────
    # 6. AUC Marginal
    # ─────────────────────────────────────────────────────────────────────────

    def _marginal_auc(self, feature_name: str, signal: pd.Series) -> dict:
        """
        Entraîne LightGBM avec et sans le signal — mesure ΔAUC.
        Source : Man AHL — "marginal contribution test" avant tout ajout en production.
        """
        from src.models.signal_generator import SignalGenerator

        sig_gen = SignalGenerator()
        target  = sig_gen._build_target(self.df)

        # Dédupliquer tout avant alignement
        feat_clean   = self.features[~self.features.index.duplicated(keep='first')]
        target_clean = target[~target.index.duplicated(keep='first')]
        signal_clean = signal[~signal.index.duplicated(keep='first')]

        common = feat_clean.index.intersection(target_clean.index).intersection(signal_clean.index)
        X_base = feat_clean.loc[common].fillna(0)
        y      = target_clean.loc[common]

        # Dataset augmenté (+ signal)
        sig_aligned = signal_clean.reindex(common).fillna(0)
        X_aug       = X_base.copy()
        X_aug[feature_name] = sig_aligned

        # Walk-forward 70/30
        split = int(len(X_base) * 0.70)
        params = {
            "objective": "binary", "metric": "auc", "n_estimators": 300,
            "learning_rate": 0.05, "num_leaves": 31, "verbose": -1,
            "n_jobs": -1, "random_state": 42,
        }

        results = {}
        for label, X in [("base", X_base), ("augmented", X_aug)]:
            X_tr, X_te = X.iloc[:split], X.iloc[split:]
            y_tr, y_te = y.iloc[:split], y.iloc[split:]
            m = lgb.LGBMClassifier(**params)
            m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            proba = m.predict_proba(X_te)[:, 1]
            results[label] = roc_auc_score(y_te, proba)

        delta = results["augmented"] - results["base"]
        return {
            "auc_base":      round(results["base"], 4),
            "auc_augmented": round(results["augmented"], 4),
            "delta_auc":     round(delta, 4),
            "improves":      delta >= AUC_DELTA_MIN,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Rapport final
    # ─────────────────────────────────────────────────────────────────────────

    def _print_report(self, feature_name: str, results: dict):
        """Affiche le rapport complet avec verdict final."""
        r      = results
        ic     = r["ic_mean"]
        icir   = r["icir"]
        tstat  = r["tstat"]
        orth   = r["orthogonality"]
        decay  = r["alpha_decay"]
        yearly = r["yearly_ic"]
        regime = r["regime_ic"]
        auc    = r.get("marginal_auc", {})

        # Verdict
        issues = []
        warns  = []

        if abs(ic) < IC_MIN:
            issues.append(f"IC trop faible ({ic:.4f} < {IC_MIN})")
        if abs(icir) < ICIR_MIN:
            warns.append(f"ICIR limite ({icir:.3f} < {ICIR_MIN})")
        if abs(tstat) < TSTAT_MIN:
            issues.append(f"Non significatif (t={tstat:.2f} < {TSTAT_MIN})")
        if orth["is_redundant"]:
            issues.append(f"Redondant avec '{orth['max_feat']}' (corr={orth['max_corr']:.2f})")
        if auc and not auc["improves"]:
            warns.append(f"AUC marginal faible (Δ={auc['delta_auc']:+.4f})")

        # Horizon naturel du signal
        max_h = max(decay, key=lambda h: abs(decay[h]))
        if max_h != self.horizon and abs(decay.get(self.horizon, 0)) < abs(decay[max_h]) * DECAY_RATIO_MIN:
            warns.append(f"Horizon naturel H+{max_h} ≠ notre horizon H+{self.horizon}")

        if len(issues) == 0 and len(warns) == 0:
            verdict = "✅ ACCEPT"
        elif len(issues) == 0:
            verdict = "⚠️  CONDITIONAL"
        elif len(issues) <= 1 and len(warns) == 0:
            verdict = "⚠️  CONDITIONAL"
        else:
            verdict = "❌ REJECT"

        logger.info("")
        logger.info("═" * 60)
        logger.info(f"  SIGNAL VALIDATOR — {feature_name.upper()}")
        logger.info("═" * 60)

        # IC
        logger.info(f"\n  1. IC Analysis (Spearman, horizon H+{self.horizon})")
        logger.info(f"     IC moyen   : {ic:+.4f}  {'✅' if abs(ic) >= IC_MIN else '❌'} (min {IC_MIN})")
        logger.info(f"     ICIR       : {icir:+.3f}  {'✅' if abs(icir) >= ICIR_MIN else '⚠️'} (min {ICIR_MIN})")
        logger.info(f"     t-stat     : {tstat:+.2f}  {'✅' if abs(tstat) >= TSTAT_MIN else '❌'} (min {TSTAT_MIN})")

        # Alpha Decay
        logger.info(f"\n  2. Alpha Decay (IC par horizon)")
        decay_str = "  ".join([f"H+{h}={v:+.3f}" for h, v in sorted(decay.items())])
        logger.info(f"     {decay_str}")
        logger.info(f"     → Horizon naturel : H+{max_h}")

        # Orthogonalité
        logger.info(f"\n  3. Orthogonalité")
        logger.info(f"     Max corrélation    : {orth['max_corr']:.3f} avec '{orth['max_feat']}'"
                    f"  {'❌ REDONDANT' if orth['is_redundant'] else '✅ OK'}")
        if orth["high_corrs"]:
            for feat, corr in list(orth["high_corrs"].items())[:3]:
                logger.info(f"     └─ {feat:30s} : {corr:.3f}")

        # Régimes
        if regime:
            logger.info(f"\n  4. IC par régime HMM")
            for name, data in regime.items():
                if data:
                    logger.info(f"     {name:8s} : IC={data['ic']:+.4f}  n={data['n']}  p={data['pval']:.3f}")

        # Annuel
        if yearly:
            logger.info(f"\n  5. IC par année (stress test)")
            positive_years = sum(1 for v in yearly.values() if v > 0)
            for year, ic_yr in yearly.items():
                icon = "✅" if ic_yr > 0.01 else ("⚠️ " if ic_yr >= 0 else "❌")
                logger.info(f"     {year} : {ic_yr:+.4f}  {icon}")
            logger.info(f"     → {positive_years}/{len(yearly)} années positives")

        # AUC marginal
        if auc:
            logger.info(f"\n  6. AUC Marginal")
            logger.info(f"     Base      : {auc['auc_base']:.4f}")
            logger.info(f"     Augmenté  : {auc['auc_augmented']:.4f}")
            logger.info(f"     Δ AUC     : {auc['delta_auc']:+.4f}  {'✅' if auc['improves'] else '⚠️'}")

        # Issues & warns
        if issues or warns:
            logger.info(f"\n  Problèmes détectés :")
            for issue in issues:
                logger.warning(f"     ❌ {issue}")
            for warn in warns:
                logger.warning(f"     ⚠️  {warn}")

        logger.info(f"\n  {'─'*56}")
        logger.info(f"  VERDICT : {verdict}")
        logger.info(f"  {'─'*56}\n")
        logger.info("═" * 60)

        return verdict

    # ─────────────────────────────────────────────────────────────────────────
    # Point d'entrée principal
    # ─────────────────────────────────────────────────────────────────────────

    def validate(
        self,
        feature_name: str,
        signal: Optional[pd.Series] = None,
        skip_auc: bool = False,
    ) -> str:
        """
        Valide une feature et retourne le verdict.

        Args:
            feature_name : nom de la feature (doit exister dans les features
                           calculées OU être passée via `signal`)
            signal       : si None, cherche dans les features existantes
            skip_auc     : si True, saute le test AUC marginal (plus rapide)

        Returns:
            "ACCEPT" | "CONDITIONAL" | "REJECT"
        """
        self._load()

        # Récupérer le signal
        if signal is None:
            if feature_name not in self.features.columns:
                logger.error(f"Feature '{feature_name}' introuvable. Passe-la via signal=pd.Series(...)")
                return "REJECT"
            signal = self.features[feature_name]

        signal = signal[~signal.index.duplicated(keep='first')].dropna()

        if len(signal) < 200:
            logger.error(f"Signal trop court ({len(signal)} points < 200 requis)")
            return "REJECT"

        logger.info(f"\nValidation de '{feature_name}' — {len(signal)} observations")

        # 1. IC principal
        ic_series = self._compute_ic(signal, self.horizon)
        ic_mean   = float(ic_series.mean()) if len(ic_series) > 0 else 0.0
        ic_std    = float(ic_series.std())  if len(ic_series) > 0 else 1.0
        icir      = ic_mean / ic_std if ic_std > 0 else 0.0
        tstat     = ic_mean / (ic_std / np.sqrt(len(ic_series))) if ic_std > 0 else 0.0

        # 2-6. Autres analyses
        decay  = self._alpha_decay(signal)
        orth   = self._orthogonality(signal)
        regime = self._regime_ic(signal)
        yearly = self._yearly_ic(signal)
        auc    = self._marginal_auc(feature_name, signal) if not skip_auc else {}

        results = {
            "feature":      feature_name,
            "ic_mean":      round(ic_mean, 4),
            "icir":         round(icir, 3),
            "tstat":        round(tstat, 2),
            "n_obs":        len(signal),
            "alpha_decay":  decay,
            "orthogonality": orth,
            "regime_ic":    regime,
            "yearly_ic":    yearly,
            "marginal_auc": auc,
        }

        verdict = self._print_report(feature_name, results)

        # Sauvegarde JSON
        out_path = Path(__file__).parent / f"validation_{feature_name}.json"
        with open(out_path, "w") as f:
            json.dump({**results, "verdict": verdict}, f, indent=2, default=str)
        logger.debug(f"Rapport sauvegardé → {out_path}")

        return verdict


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stdout, level="INFO", colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
    )

    features_to_validate = sys.argv[1:] if len(sys.argv) > 1 else ["atr_ratio"]

    validator = SignalValidator(horizon=4)

    for feat in features_to_validate:
        skip = "--fast" in sys.argv
        validator.validate(feat, skip_auc=skip)
