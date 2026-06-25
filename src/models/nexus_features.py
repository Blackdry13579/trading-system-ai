"""
NEXUS Features Bridge — Traduit les scores NEXUS en features numériques pour LightGBM.

Les 4 couches NEXUS existent dans nexus.py mais leurs scores ne sont pas
encore utilisés comme features d'entraînement. Ce module fait le pont.

Architecture :
  nexus.py (scores 0-100) → nexus_features.py (features normalisées) → feature_builder.py
"""

import pandas as pd
import numpy as np
from loguru import logger


def compute_nexus_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les 4 scores NEXUS à partir du DataFrame OHLCV enrichi.

    Ces scores sont calculés directement ici (sans importer nexus.py)
    pour éviter le cycle d'import identifié par graphify.

    Returns:
        DataFrame original avec colonnes nexus_* ajoutées.
    """
    result = df.copy()

    # ── Couche 1 — Macro Bias (25%) ──────────────────────────────────────────
    # Signal : real rates négatifs + COT bullish + dollar faible = bullish gold
    macro_score = 50.0  # neutre par défaut

    if "real_rate_signal" in df.columns:
        real_rate_component = df["real_rate_signal"].iloc[-1]  # -1 ou +1
        macro_score += real_rate_component * 15  # ±15 points

    if "cot_signal_num" in df.columns:
        cot_component = df["cot_signal_num"].iloc[-1]  # -1, 0, +1
        macro_score += cot_component * 10  # ±10 points

    if "fred_DTWEXBGS" in df.columns:  # Dollar index
        dxy_val = df["fred_DTWEXBGS"].iloc[-1]
        dxy_pct_change = df["fred_DTWEXBGS"].pct_change(5).iloc[-1] if len(df) > 5 else 0
        dxy_component = -1.0 if dxy_pct_change > 0.005 else (1.0 if dxy_pct_change < -0.005 else 0.0)
        macro_score += dxy_component * 5  # ±5 points

    result["nexus_macro_score"] = round(max(0, min(100, macro_score)), 1)

    # ── Couche 2 — Flux Institutionnels (35%) ────────────────────────────────
    # Signal : ETF inflows + sentiment contrarian = pression haussière
    flux_score = 50.0

    if "etf_combined_flow" in df.columns:
        etf_flow = df["etf_combined_flow"].iloc[-1]
        # >1.5 = flux entrants forts → bullish | <0.7 = sorties → bearish
        etf_component = 1.0 if etf_flow > 1.5 else (-1.0 if etf_flow < 0.7 else 0.0)
        flux_score += etf_component * 20  # ±20 points

    if "sentiment_contrarian" in df.columns:
        contrarian = df["sentiment_contrarian"].iloc[-1]  # -1 à +1
        flux_score += contrarian * 15  # ±15 points

    if "nim_sentiment" in df.columns:
        nim = df["nim_sentiment"].iloc[-1]  # -1 à +1
        flux_score += nim * 10  # ±10 points

    result["nexus_flux_score"] = round(max(0, min(100, flux_score)), 1)

    # ── Couche 3 — Régime & Timing (40%) ─────────────────────────────────────
    # Signal : multi-timeframe confluence + alpha decay
    regime_score = 50.0

    # EMA trend alignment (1h vs 4h proxy via rolling windows)
    if "close" in df.columns and len(df) >= 96:
        close = df["close"]
        ema_20   = close.ewm(span=20).mean()
        ema_50   = close.ewm(span=50).mean()
        ema_200  = close.ewm(span=200).mean()
        ema_96   = close.ewm(span=96).mean()   # proxy 4h (96 × 1h bougies)

        last_close = float(close.iloc[-1])
        aligned_bull = (last_close > float(ema_20.iloc[-1]) >
                        float(ema_50.iloc[-1]) > float(ema_200.iloc[-1]))
        aligned_bear = (last_close < float(ema_20.iloc[-1]) <
                        float(ema_50.iloc[-1]) < float(ema_200.iloc[-1]))

        if aligned_bull:
            regime_score += 20
        elif aligned_bear:
            regime_score -= 20

        # 4h confirmation
        above_4h_ema = last_close > float(ema_96.iloc[-1])
        regime_score += 10 if above_4h_ema else -10

    # London session bonus (signal de qualité supérieure)
    from datetime import datetime, timezone
    hour = datetime.now(timezone.utc).hour
    in_london = 8 <= hour < 17
    in_ny     = 13 <= hour < 22
    if in_london and in_ny:   # overlap = meilleure liquidité
        regime_score += 10
    elif in_london or in_ny:
        regime_score += 5

    result["nexus_regime_score"] = round(max(0, min(100, regime_score)), 1)

    # ── Score Global NEXUS (pondéré) ──────────────────────────────────────────
    nexus_global = (
        result["nexus_macro_score"].iloc[-1]  * 0.25 +
        result["nexus_flux_score"].iloc[-1]   * 0.35 +
        result["nexus_regime_score"].iloc[-1] * 0.40
    )
    result["nexus_global_score"] = round(nexus_global, 1)

    # Signal directionnel normalisé (-1 à +1) pour LightGBM
    result["nexus_direction"] = round((nexus_global - 50) / 50, 3)

    logger.debug(
        f"  NEXUS : macro={result['nexus_macro_score'].iloc[-1]:.0f} "
        f"flux={result['nexus_flux_score'].iloc[-1]:.0f} "
        f"regime={result['nexus_regime_score'].iloc[-1]:.0f} "
        f"global={nexus_global:.0f}"
    )

    return result


# ── Test standalone ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import yfinance as yf
    logger.remove()
    logger.add(sys.stdout, level="DEBUG", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    logger.info("Test NEXUS features...")
    df = yf.download("GC=F", period="30d", interval="1h",
                     progress=False, auto_adjust=True)
    df.columns = [c.lower() for c in df.columns.get_level_values(0)]

    df_enriched = compute_nexus_features(df)

    nexus_cols = [c for c in df_enriched.columns if "nexus" in c]
    logger.success(f"Features NEXUS ajoutées : {nexus_cols}")
    logger.info(df_enriched[nexus_cols].iloc[-1].to_string())
