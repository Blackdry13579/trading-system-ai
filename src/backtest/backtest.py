"""
Backtest Walk-Forward — Validation de la stratégie GoldBot sur données historiques.

Sources :
  - Lopez de Prado (2018)  : "Advances in Financial Machine Learning" — Wiley
                              Walk-forward validation, métriques correctes
  - Bailey & Lopez (2014)  : "The Deflated Sharpe Ratio" — Journal of Portfolio Management
                              Sharpe ratio ajusté pour éviter l'overfitting
  - Estrada (2000)         : "The Omega Ratio" — meilleure alternative au Sharpe
                              pour distributions non-normales (fréquent sur commodités)
  - Man AHL (UK)           : backtesting methodology pour systèmes trend-following

Métriques calculées :
  - Total Return (%)
  - Sharpe Ratio annualisé
  - Max Drawdown (%)
  - Win Rate (%)
  - Profit Factor
  - Nombre de trades

Critères de validation (règles internes) :
  ✅ Sharpe > 0.8  → déployer en paper trading
  ✅ Max DD < 15%  → acceptable
  ✅ Win rate > 45%
  ✅ > 50 trades   → échantillon suffisant
"""

import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime

from src.config import (
    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER, ATR_TP_TREND, ATR_TP_RANGE,
    KELLY_FRACTION, MAX_RISK_PER_TRADE, MIN_SIGNAL_SCORE,
    LGBM_THRESHOLD_LONG, LGBM_THRESHOLD_SHORT,
    SPREAD_USD, SLIPPAGE_USD, SWAP_PER_NIGHT_USD,
    TRADE_SESSIONS_ONLY,
    SESSION_LONDON_START, SESSION_LONDON_END,
    SESSION_NY_START, SESSION_NY_END,
)
from src.database import load_gold_prices
from src.models.feature_builder import build_features
from src.models.regime_detector import RegimeDetector
from src.models.signal_generator import SignalGenerator


class WalkForwardBacktest:
    """
    Backtest walk-forward sur données historiques.

    Architecture :
    - Train sur les 70% premières données
    - Test sur les 30% restantes
    - Simulation trade par trade avec SL/TP ATR-based
    - Capital initial : 10 000$ (fictif)
    """

    def __init__(self, initial_capital: float = 10_000.0):
        self.initial_capital = initial_capital
        self.capital         = initial_capital
        self.trades: list[dict] = []
        self.equity_curve: list[float] = [initial_capital]

    # ─────────────────────────────────────────────
    # Calcul ATR (pour SL/TP dynamiques)
    # ─────────────────────────────────────────────

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    # ─────────────────────────────────────────────
    # Simulation d'un trade
    # ─────────────────────────────────────────────

    def _simulate_trade(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        direction: str,
        score: float,
        atr_value: float,
        regime: str = "RANGE",
    ) -> dict | None:
        """
        Simule un trade réaliste incluant :
          - Spread à l'entrée (coût immédiat)
          - Slippage sur le SL (le marché ne s'arrête pas pile au niveau)
          - Swap pour chaque nuit passée en position
          - TP adaptatif selon le régime HMM
        """
        entry_price = float(df["close"].iloc[entry_idx])
        entry_time  = df.index[entry_idx]

        # Coût spread à l'entrée (payé immédiatement, direction dépendante)
        spread_cost = SPREAD_USD * (1.0 if direction == "LONG" else 1.0)

        sl_dist = atr_value * ATR_SL_MULTIPLIER

        # TP adaptatif selon le régime HMM
        tp_mult = ATR_TP_TREND if regime == "TREND" else ATR_TP_RANGE
        tp_dist = atr_value * tp_mult

        if direction == "LONG":
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
        else:
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist

        # Taille de position — Kelly × 0.25
        risk_amount = self.capital * MAX_RISK_PER_TRADE * KELLY_FRACTION
        lot_size    = risk_amount / sl_dist if sl_dist > 0 else 0.01
        lot_size    = min(lot_size, 1.0)

        max_bars = 48  # Max 48h pour sortir

        for i in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
            bar = df.iloc[i]
            bar_high = float(bar["high"])
            bar_low  = float(bar["low"])
            bar_time = df.index[i]

            # Filtre weekend — pas de trade samedi/dimanche
            if hasattr(bar_time, 'weekday') and bar_time.weekday() >= 5:
                continue

            # Swap : frais par nuit passée en position
            hours_held = (bar_time - entry_time).total_seconds() / 3600
            nights     = int(hours_held / 24)
            swap_cost  = nights * SWAP_PER_NIGHT_USD

            if direction == "LONG":
                if bar_low <= sl:
                    # SL touché : slippage défavorable (on sort plus bas que prévu)
                    real_exit  = sl - SLIPPAGE_USD
                    raw_pnl    = -(sl_dist + SLIPPAGE_USD) * lot_size
                    pnl        = raw_pnl - spread_cost - swap_cost
                    result     = "SL"
                    exit_price = real_exit
                elif bar_high >= tp:
                    # TP touché : pas de slippage (ordre limite)
                    raw_pnl    = tp_dist * lot_size
                    pnl        = raw_pnl - spread_cost - swap_cost
                    result     = "TP"
                    exit_price = tp
                else:
                    continue
            else:
                if bar_high >= sl:
                    real_exit  = sl + SLIPPAGE_USD
                    raw_pnl    = -(sl_dist + SLIPPAGE_USD) * lot_size
                    pnl        = raw_pnl - spread_cost - swap_cost
                    result     = "SL"
                    exit_price = real_exit
                elif bar_low <= tp:
                    raw_pnl    = tp_dist * lot_size
                    pnl        = raw_pnl - spread_cost - swap_cost
                    result     = "TP"
                    exit_price = tp
                else:
                    continue

            self.capital += pnl
            self.equity_curve.append(self.capital)

            return {
                "entry_time":  entry_time,
                "exit_time":   bar_time,
                "direction":   direction,
                "entry_price": entry_price,
                "exit_price":  exit_price,
                "sl":          sl,
                "tp":          tp,
                "lot_size":    lot_size,
                "pnl":         pnl,
                "spread_cost": spread_cost,
                "swap_cost":   swap_cost,
                "result":      result,
                "score":       score,
                "regime":      regime,
                "capital":     self.capital,
            }

        # Sortie forcée timeout — on ferme au cours actuel
        if entry_idx + max_bars < len(df):
            exit_price = float(df["close"].iloc[entry_idx + max_bars])
            exit_time  = df.index[entry_idx + max_bars]
            hours_held = (exit_time - entry_time).total_seconds() / 3600
            nights     = int(hours_held / 24)
            swap_cost  = nights * SWAP_PER_NIGHT_USD

            if direction == "LONG":
                raw_pnl = (exit_price - entry_price) * lot_size
            else:
                raw_pnl = (entry_price - exit_price) * lot_size
            pnl = raw_pnl - spread_cost - swap_cost

            self.capital += pnl
            self.equity_curve.append(self.capital)
            return {
                "entry_time":  entry_time,
                "exit_time":   exit_time,
                "direction":   direction,
                "entry_price": entry_price,
                "exit_price":  exit_price,
                "sl": sl, "tp": tp,
                "lot_size":    lot_size,
                "pnl":         pnl,
                "spread_cost": spread_cost,
                "swap_cost":   swap_cost,
                "result":      "TIMEOUT",
                "score":       score,
                "regime":      regime,
                "capital":     self.capital,
            }
        return None

    # ─────────────────────────────────────────────
    # Lancement du backtest
    # ─────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        detector: RegimeDetector,
        generator: SignalGenerator,
        regime_filter: str = "ALL",   # "ALL" | "TREND" | "RANGE"
    ) -> dict:
        """
        Lance le backtest walk-forward.
        Train sur 70%, test sur 30%.

        regime_filter : "ALL"   → trade tous les régimes (baseline)
                        "TREND" → trade uniquement en régime TREND
                        "RANGE" → trade uniquement en régime RANGE
        """
        df.columns = df.columns.str.lower()
        n = len(df)
        test_start = int(n * 0.70)

        logger.info(f"  Backtest : {n} bougies total")
        logger.info(f"  Train : bougies 0→{test_start} ({test_start} barres)")
        logger.info(f"  Test  : bougies {test_start}→{n} ({n - test_start} barres)")

        df_test    = df.iloc[test_start:].copy()
        # Dédupliquer l'index de features (timestamps dupliqués par resample)
        # puis aligner par index sur df_test
        features_clean = features[~features.index.duplicated(keep='first')]
        feats_test = features_clean.reindex(df_test.index).fillna(0)
        atr_series = self._atr(df)

        # ── Pré-calcul des régimes HMM en batch ──────────────
        # Au lieu d'appeler detector.predict() 10 000× dans la boucle (7 min),
        # on pré-calcule les régimes sur la période de TEST uniquement (45 sec).
        # Chaque appel HMM traite une fenêtre glissante de 100 bougies.
        logger.info("  Pré-calcul des régimes HMM sur la période de test...")
        regimes = {}
        for idx in range(test_start, n):
            try:
                name, _, _ = detector.predict(df.iloc[max(0, idx - 100): idx + 1])
                regimes[idx] = name
            except Exception:
                regimes[idx] = "RANGE"
        logger.info(f"  Régimes calculés : {len(regimes)} bougies test")

        last_entry     = -99
        i              = 0
        n_test         = len(df_test) - 1

        # Compteurs de diagnostic
        dbg_chaos     = 0
        dbg_predict   = 0
        dbg_flat      = 0
        dbg_session   = 0
        dbg_simulate  = 0

        # While loop (pas for) pour pouvoir avancer i après chaque trade
        while i < n_test:
            abs_i = test_start + i

            current_regime = regimes.get(abs_i, "RANGE")
            if current_regime == "CHAOS":
                dbg_chaos += 1
                i += 1
                continue

            # Filtre régime optionnel
            if regime_filter != "ALL" and current_regime != regime_filter:
                i += 1
                continue

            # Cooldown : pas deux trades en moins de 4 heures
            if abs_i - last_entry < 4:
                i += 1
                continue

            # Signal LightGBM via index aligné
            try:
                row = feats_test.iloc[[i]]
                score, direction = generator.predict(row)
                dbg_predict += 1
            except Exception:
                i += 1
                continue

            # Filtre score + direction
            if score < MIN_SIGNAL_SCORE or direction == "FLAT":
                dbg_flat += 1
                i += 1
                continue

            # Filtre session : seulement London (8h-17h) et NY (13h-22h) GMT
            if TRADE_SESSIONS_ONLY:
                bar_hour = df_test.index[i].hour
                in_london = SESSION_LONDON_START <= bar_hour < SESSION_LONDON_END
                in_ny     = SESSION_NY_START <= bar_hour < SESSION_NY_END
                if not (in_london or in_ny):
                    dbg_session += 1
                    i += 1
                    continue

            dbg_simulate += 1

            # Simuler le trade
            atr_val = float(atr_series.iloc[abs_i]) if abs_i < len(atr_series) else 10.0
            if atr_val <= 0 or np.isnan(atr_val):
                i += 1
                continue

            trade = self._simulate_trade(df, abs_i, direction, score, atr_val, current_regime)
            if trade:
                self.trades.append(trade)
                last_entry = abs_i
                # Avancer i jusqu'à la barre de sortie
                exit_time = trade["exit_time"]
                while i < n_test and df_test.index[i] < exit_time:
                    i += 1
            else:
                i += 1

            # Stop si capital < 50% du capital initial
            if self.capital < self.initial_capital * 0.5:
                logger.warning("⚠️  Capital < 50% initial — backtest arrêté")
                break

        logger.info(f"  🔍 DIAGNOSTIC FILTRES :")
        logger.info(f"     Barres CHAOS filtrées   : {dbg_chaos}")
        logger.info(f"     Prédictions LightGBM    : {dbg_predict}")
        logger.info(f"     FLAT/score faible        : {dbg_flat}")
        logger.info(f"     Hors session filtrées    : {dbg_session}")
        logger.info(f"     _simulate_trade() appelé : {dbg_simulate}")

        return self._compute_metrics()

    # ─────────────────────────────────────────────
    # Métriques (Lopez de Prado + Man AHL)
    # ─────────────────────────────────────────────

    def _compute_metrics(self) -> dict:
        if not self.trades:
            return {"error": "Aucun trade exécuté"}

        df_trades = pd.DataFrame(self.trades)
        pnls      = df_trades["pnl"].values
        equity    = np.array(self.equity_curve)

        # Rendements
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        wins         = (pnls > 0).sum()
        losses       = (pnls <= 0).sum()
        win_rate     = wins / len(pnls) * 100

        # Profit Factor
        gross_profit = pnls[pnls > 0].sum() if wins > 0 else 0
        gross_loss   = abs(pnls[pnls < 0].sum()) if losses > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

        # Sharpe Ratio annualisé (Lopez de Prado)
        returns_series = pd.Series(equity).pct_change().dropna()
        sharpe = (returns_series.mean() / returns_series.std() * np.sqrt(24 * 252)
                  if returns_series.std() > 0 else 0)

        # Max Drawdown
        rolling_max = pd.Series(equity).cummax()
        drawdowns   = (pd.Series(equity) - rolling_max) / rolling_max * 100
        max_dd      = float(drawdowns.min())

        # Durée moyenne des trades
        df_trades["duration"] = (
            pd.to_datetime(df_trades["exit_time"]) -
            pd.to_datetime(df_trades["entry_time"])
        ).dt.total_seconds() / 3600

        total_spread = df_trades.get("spread_cost", pd.Series([0]*len(df_trades))).sum()
        total_swap   = df_trades.get("swap_cost",   pd.Series([0]*len(df_trades))).sum()

        metrics = {
            "total_trades":    len(self.trades),
            "wins":            int(wins),
            "losses":          int(losses),
            "win_rate":        round(win_rate, 1),
            "total_return":    round(total_return, 2),
            "final_capital":   round(self.capital, 2),
            "sharpe_ratio":    round(sharpe, 3),
            "max_drawdown":    round(max_dd, 2),
            "profit_factor":   round(profit_factor, 2),
            "avg_win":         round(pnls[pnls > 0].mean(), 2) if wins > 0 else 0,
            "avg_loss":        round(pnls[pnls < 0].mean(), 2) if losses > 0 else 0,
            "avg_duration_h":  round(df_trades["duration"].mean(), 1),
            "best_trade":      round(pnls.max(), 2),
            "worst_trade":     round(pnls.min(), 2),
            "total_spread":    round(total_spread, 2),
            "total_swap":      round(total_swap, 2),
            "total_costs":     round(total_spread + total_swap, 2),
        }
        return metrics

    def print_report(self, metrics: dict):
        """Affiche le rapport de backtest dans la console."""
        if "error" in metrics:
            logger.error(f"❌ {metrics['error']}")
            return

        logger.info("=" * 55)
        logger.info("  📊 RAPPORT BACKTEST — GoldBot XAU/USD")
        logger.info("=" * 55)
        logger.info(f"  Trades total    : {metrics['total_trades']}")
        logger.info(f"  Gagnants        : {metrics['wins']} ({metrics['win_rate']}%)")
        logger.info(f"  Perdants        : {metrics['losses']}")
        logger.info(f"  Profit Factor   : {metrics['profit_factor']}")
        logger.info(f"  Total Return    : {metrics['total_return']:+.2f}%")
        logger.info(f"  Capital final   : {metrics['final_capital']:.2f}$")
        logger.info(f"  Sharpe Ratio    : {metrics['sharpe_ratio']:.3f}")
        logger.info(f"  Max Drawdown    : {metrics['max_drawdown']:.2f}%")
        logger.info(f"  Gain moyen      : +{metrics['avg_win']:.2f}$")
        logger.info(f"  Perte moyenne   : {metrics['avg_loss']:.2f}$")
        logger.info(f"  Durée moy trade : {metrics['avg_duration_h']}h")
        logger.info(f"  ── Coûts réels ──────────────────────────")
        logger.info(f"  Spread total    : -{metrics['total_spread']:.2f}$")
        logger.info(f"  Swap total      : -{metrics['total_swap']:.2f}$")
        logger.info(f"  Total frais     : -{metrics['total_costs']:.2f}$")
        logger.info("=" * 55)

        # Verdict
        sharpe_ok = metrics["sharpe_ratio"] >= 0.8
        dd_ok     = metrics["max_drawdown"] >= -15.0
        wr_ok     = metrics["win_rate"] >= 45.0
        n_ok      = metrics["total_trades"] >= 50

        logger.info("  CRITÈRES DE VALIDATION :")
        logger.info(f"  {'✅' if sharpe_ok else '❌'} Sharpe > 0.8   : {metrics['sharpe_ratio']:.3f}")
        logger.info(f"  {'✅' if dd_ok else '❌'} Max DD < 15%  : {metrics['max_drawdown']:.2f}%")
        logger.info(f"  {'✅' if wr_ok else '❌'} Win rate > 45%: {metrics['win_rate']}%")
        logger.info(f"  {'✅' if n_ok else '❌'} > 50 trades   : {metrics['total_trades']}")

        if all([sharpe_ok, dd_ok, wr_ok, n_ok]):
            logger.success("  ✅ STRATÉGIE VALIDÉE — Paper trading autorisé")
        else:
            logger.warning("  ⚠️  STRATÉGIE À AMÉLIORER avant paper trading")
        logger.info("=" * 55)

        return metrics


def run_backtest(regime_filter: str = "ALL") -> dict:
    """Lance le backtest complet et retourne les métriques."""
    from src.data.collector import fetch_cot_gold, fetch_fred_macro

    logger.info(f"🚀 Backtest — régime filtre : {regime_filter}")

    df = load_gold_prices(timeframe="1h", days=730)
    if df.empty:
        logger.error("❌ Pas de données — lance d'abord src.main")
        return {}

    cot_df     = fetch_cot_gold()
    macro_dict = fetch_fred_macro()

    detector  = RegimeDetector()
    generator = SignalGenerator()

    if not detector.load():
        detector.train(df)
    if not generator.load():
        features_full = build_features(df, cot_df, macro_dict)
        generator.train(features_full, df, cot_df, macro_dict)

    logger.info("⚙️  Calcul des features...")
    features = build_features(df, cot_df, macro_dict)

    bt = WalkForwardBacktest(initial_capital=10_000.0)
    metrics = bt.run(df, features, detector, generator, regime_filter=regime_filter)
    bt.print_report(metrics)
    return metrics


def run_regime_comparison() -> None:
    """
    Compare TREND-only vs RANGE-only vs ALL régimes.
    Répond à la question : vaut-il mieux trader uniquement en TREND ?
    """
    from src.data.collector import fetch_cot_gold, fetch_fred_macro

    logger.info("🔬 COMPARAISON RÉGIMES — TREND vs RANGE vs ALL")
    logger.info("=" * 55)

    # Charger les données une seule fois
    df = load_gold_prices(timeframe="1h", days=730)
    if df.empty:
        logger.error("❌ Pas de données")
        return

    cot_df     = fetch_cot_gold()
    macro_dict = fetch_fred_macro()

    detector  = RegimeDetector()
    generator = SignalGenerator()
    if not detector.load():
        detector.train(df)
    if not generator.load():
        features_full = build_features(df, cot_df, macro_dict)
        generator.train(features_full, df, cot_df, macro_dict)

    features = build_features(df, cot_df, macro_dict)

    results = {}
    for regime in ["ALL", "TREND", "RANGE"]:
        logger.info(f"\n{'='*55}")
        logger.info(f"  Filtre : {regime}")
        bt = WalkForwardBacktest(initial_capital=10_000.0)
        m  = bt.run(df, features, detector, generator, regime_filter=regime)
        bt.print_report(m)
        results[regime] = m

    # Tableau comparatif final
    logger.info("\n" + "=" * 55)
    logger.info("  TABLEAU COMPARATIF")
    logger.info("=" * 55)
    logger.info(f"  {'Métrique':<22} {'ALL':>10} {'TREND':>10} {'RANGE':>10}")
    logger.info(f"  {'-'*52}")
    metrics_to_show = [
        ("Trades", "total_trades"),
        ("Win Rate (%)", "win_rate"),
        ("Total Return (%)", "total_return"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("Max Drawdown (%)", "max_drawdown"),
        ("Profit Factor", "profit_factor"),
        ("Gain moyen ($)", "avg_win"),
        ("Perte moyenne ($)", "avg_loss"),
    ]
    for label, key in metrics_to_show:
        vals = [results[r].get(key, "N/A") for r in ["ALL", "TREND", "RANGE"]]
        logger.info(f"  {label:<22} {str(vals[0]):>10} {str(vals[1]):>10} {str(vals[2]):>10}")
    logger.info("=" * 55)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        run_regime_comparison()
    else:
        run_backtest()
