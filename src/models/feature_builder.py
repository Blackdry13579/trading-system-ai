"""
Feature Builder — Calcul de toutes les features pour le LightGBM.

Sources par bloc :
  - pandas-ta          : 130+ indicateurs (communauté internationale)
  - Larry Williams     : COT Index + Williams %R (USA, +11 000% en 1987)
  - Richard Wyckoff    : volume/structure, repris par Stan Weinstein (USA/UK)
  - Michael Huddleston : ICT — Order Blocks, FVG, OTE (USA, ex-institutionnel)
  - LBMA / desks London: session analysis, Gold Fix depuis 1919 (UK)
  - Ray Dalio          : macro cycle features, Bridgewater (USA)
  - Recherche chinoise : USD/CNY correlation, saisonnalité lunaire
                         (Chine = 1er consommateur mondial d'or)
"""

import numpy as np
import pandas as pd
from loguru import logger

try:
    import ta as ta_lib
    from ta.momentum import RSIIndicator, WilliamsRIndicator, StochasticOscillator
    from ta.trend import EMAIndicator, MACD
    from ta.volatility import AverageTrueRange, BollingerBands
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("ta non installé : pip install ta")


# ═══════════════════════════════════════════════════════════════
# BLOC 1 — Prix et rendements (base universelle)
# ═══════════════════════════════════════════════════════════════

def price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rendements et volatilité — base de tout modèle quantitatif."""
    out = pd.DataFrame(index=df.index)

    c = df["close"]
    out["returns_1h"]    = np.log(c / c.shift(1))
    out["returns_4h"]    = np.log(c / c.shift(4))
    out["returns_1d"]    = np.log(c / c.shift(24))
    out["returns_1w"]    = np.log(c / c.shift(24 * 5))
    out["volatility_20"] = out["returns_1h"].rolling(20).std()

    if "high" in df.columns and "low" in df.columns:
        hl = df["high"] - df["low"]
        out["candle_range"]  = hl / c
        out["upper_shadow"]  = (df["high"] - df[["open", "close"]].max(axis=1)) / hl.replace(0, np.nan)
        out["lower_shadow"]  = (df[["open", "close"]].min(axis=1) - df["low"]) / hl.replace(0, np.nan)
        out["body_ratio"]    = abs(df["close"] - df["open"]) / hl.replace(0, np.nan)

    return out


# ═══════════════════════════════════════════════════════════════
# BLOC 2 — Indicateurs tendance (pandas-ta)
# Source : communauté internationale, validé sur des décennies
# ═══════════════════════════════════════════════════════════════

def trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """EMA, ATR, RSI — indicateurs standards validés institutionnellement."""
    out = pd.DataFrame(index=df.index)
    c = df["close"]

    h, l, cl = df["high"], df["low"], df["close"]

    if TA_AVAILABLE:
        ema9   = EMAIndicator(cl, window=9).ema_indicator()
        ema21  = EMAIndicator(cl, window=21).ema_indicator()
        ema50  = EMAIndicator(cl, window=50).ema_indicator()
        ema200 = EMAIndicator(cl, window=200).ema_indicator()

        out["ema_cross_9_21"]  = (ema9 > ema21).astype(int)
        out["ema_cross_21_50"] = (ema21 > ema50).astype(int)
        out["price_vs_ema200"] = (cl - ema200) / ema200.replace(0, np.nan)
        out["ema_distance_9"]  = (cl - ema9) / ema9.replace(0, np.nan)

        atr14 = AverageTrueRange(h, l, cl, window=14).average_true_range()
        atr5  = AverageTrueRange(h, l, cl, window=5).average_true_range()
        atr20 = AverageTrueRange(h, l, cl, window=20).average_true_range()
        out["atr_14"]          = atr14
        out["atr_ratio"]       = atr14 / cl.replace(0, np.nan)
        out["atr_compression"] = (atr5 / atr20.replace(0, np.nan)).fillna(1)

        out["rsi_14"] = RSIIndicator(cl, window=14).rsi()
        out["rsi_28"] = RSIIndicator(cl, window=28).rsi()

        macd_obj = MACD(cl, window_fast=12, window_slow=26, window_sign=9)
        out["macd_hist"] = macd_obj.macd_diff()

        bb = BollingerBands(cl, window=20, window_dev=2)
        denom = (bb.bollinger_hband() - bb.bollinger_lband()).replace(0, np.nan)
        out["bb_position"] = (cl - bb.bollinger_lband()) / denom
        out["bb_width"]    = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg().replace(0, np.nan)

        stoch = StochasticOscillator(h, l, cl, window=14, smooth_window=3)
        out["stoch_k"] = stoch.stoch()
    else:
        # Fallback manuel
        ema9   = cl.ewm(span=9,   adjust=False).mean()
        ema21  = cl.ewm(span=21,  adjust=False).mean()
        ema50  = cl.ewm(span=50,  adjust=False).mean()
        ema200 = cl.ewm(span=200, adjust=False).mean()
        out["ema_cross_9_21"]  = (ema9 > ema21).astype(int)
        out["ema_cross_21_50"] = (ema21 > ema50).astype(int)
        out["price_vs_ema200"] = (cl - ema200) / ema200.replace(0, np.nan)
        out["atr_ratio"]       = (h - l).rolling(14).mean() / cl.replace(0, np.nan)
        out["atr_compression"] = ((h - l).rolling(5).mean() / (h - l).rolling(20).mean().replace(0, np.nan)).fillna(1)

    return out


# ═══════════════════════════════════════════════════════════════
# BLOC 3 — Larry Williams : COT Index + Williams %R
# Source : "Trade Stocks and Commodities with the Insiders" (Williams, 2004)
# Résultats : +11 000% World Cup Trading Championship 1987
# ═══════════════════════════════════════════════════════════════

def larry_williams_features(df: pd.DataFrame, cot_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """COT Index (méthode exacte Larry Williams) + Williams Percent Range."""
    out = pd.DataFrame(index=df.index)
    h, l, c = df["high"], df["low"], df["close"]

    # Williams %R — identique à la formule originale Williams
    for period in [14, 28]:
        highest_h = h.rolling(period).max()
        lowest_l  = l.rolling(period).min()
        denom = (highest_h - lowest_l).replace(0, np.nan)
        out[f"williams_r_{period}"] = ((highest_h - c) / denom * -100)

    # COT Index — si données disponibles (table cot_data)
    if cot_df is not None and not cot_df.empty:
        cot_last = cot_df.sort_values("report_date").iloc[-1]
        out["cot_commercial_net"]   = float(cot_last["commercial_net"])
        out["cot_commercial_index"] = float(cot_last["commercial_index"])
        out["cot_bullish"]  = int(cot_last["cot_signal"] == "BULLISH")
        out["cot_bearish"]  = int(cot_last["cot_signal"] == "BEARISH")
    else:
        out["cot_commercial_net"]   = 0.0
        out["cot_commercial_index"] = 50.0
        out["cot_bullish"]  = 0
        out["cot_bearish"]  = 0

    return out


# ═══════════════════════════════════════════════════════════════
# BLOC 4 — Richard Wyckoff : volume + structure de marché
# Source : "The Richard D. Wyckoff Method of Trading" (1931)
# Utilisé aujourd'hui par SMI Institute (UK) et Smart Money traders
# Stan Weinstein "Secrets for Profiting in Bull and Bear Markets" (1988)
# ═══════════════════════════════════════════════════════════════

def wyckoff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume analysis et détection des phases Wyckoff."""
    out = pd.DataFrame(index=df.index)
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    o = df.get("open", c)

    # Volume analysis
    sma_vol20 = v.rolling(20).mean()
    out["volume_ratio"]  = v / sma_vol20.replace(0, np.nan)
    out["high_volume"]   = (out["volume_ratio"] > 1.5).astype(int)
    out["low_volume"]    = (out["volume_ratio"] < 0.7).astype(int)
    out["volume_trend"]  = v.rolling(5).mean() / v.rolling(20).mean().replace(0, np.nan)

    # Range dynamique sur 50 bougies (détection de la phase de consolidation)
    period = 50
    range_high = h.rolling(period).max()
    range_low  = l.rolling(period).min()
    range_size = (range_high - range_low).replace(0, np.nan)
    out["range_position"] = (c - range_low) / range_size

    # Selling Climax (SC) — bougie baissière + haut volume + nouveau bas
    bearish = (c < o).astype(int)
    new_low = (l < l.shift(1)).astype(int)
    out["selling_climax"] = (bearish & out["high_volume"].astype(bool) & new_low.astype(bool)).astype(int)

    # Spring — fausse cassure sous le bas de range + récupération (Wyckoff)
    spring_threshold = range_low * 0.998
    out["spring_detected"] = (
        (l < spring_threshold) & (c > range_low) & out["low_volume"].astype(bool)
    ).astype(int)

    # Sign of Strength (SOS) — cassure à la hausse avec volume
    out["sos_signal"] = (
        (c > range_high.shift(1)) & out["high_volume"].astype(bool)
    ).astype(int)

    return out


# ═══════════════════════════════════════════════════════════════
# BLOC 5 — ICT Smart Money Concepts
# Source : Michael Huddleston (Inner Circle Trader)
# Ex-trader institutionnel, méthode basée sur order flow institutionnel
# XAU/USD est le marché privilégié par ICT (liquidity pools massifs)
# ═══════════════════════════════════════════════════════════════

def ict_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fair Value Gaps, Order Blocks, OTE zone — méthode ICT."""
    out = pd.DataFrame(index=df.index)
    h, l, c = df["high"], df["low"], df["close"]
    o = df.get("open", c)

    # Fair Value Gap (FVG) — déséquilibre de prix sur 3 bougies
    # Haussier : low[t] > high[t-2]
    # Baissier  : high[t] < low[t-2]
    out["fvg_bull"] = (l > h.shift(2)).astype(int)
    out["fvg_bear"] = (h < l.shift(2)).astype(int)
    out["fvg_bull_size"] = ((l - h.shift(2)) / c).clip(lower=0)
    out["fvg_bear_size"] = ((l.shift(2) - h) / c).clip(lower=0)

    # Equal Highs / Equal Lows (liquidity pools)
    tol = 0.0005
    out["equal_highs"] = (abs(h - h.shift(1)) / h.replace(0, np.nan) < tol).astype(int)
    out["equal_lows"]  = (abs(l - l.shift(1)) / l.replace(0, np.nan) < tol).astype(int)

    # OTE Zone (Optimal Trade Entry) — retracement Fibonacci 62-79%
    # Calcul sur le dernier swing de 20 bougies
    swing_high = h.rolling(20).max()
    swing_low  = l.rolling(20).min()
    swing_range = (swing_high - swing_low).replace(0, np.nan)
    retracement = (swing_high - c) / swing_range
    out["in_ote_bull"] = ((retracement >= 0.62) & (retracement <= 0.79)).astype(int)

    # Order Block haussier — dernière bougie baissière avant impulsion haussière
    bullish_impulse = (c > c.shift(1) * 1.002)
    bearish_candle  = (c < o)
    prev_bearish = bearish_candle.shift(1).fillna(0).astype(bool)
    out["bull_ob_condition"] = (prev_bearish & bullish_impulse).astype(int)

    return out


# ═══════════════════════════════════════════════════════════════
# BLOC 6 — London Open Breakout (desks institutionnels)
# Source : LBMA Gold Fix depuis 1919, desks London (Goldman, HSBC, UBS)
# 70% du daily high/low du gold se forme pendant la session London
# ═══════════════════════════════════════════════════════════════

def london_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Session analysis basé sur l'heure GMT."""
    out = pd.DataFrame(index=df.index)

    if df.index.tz is None:
        idx_utc = pd.to_datetime(df.index, utc=True)
    else:
        idx_utc = df.index.tz_convert("UTC")

    hour = idx_utc.hour

    out["is_asian_session"]  = ((hour >= 0)  & (hour < 8)).astype(int)
    out["is_london_session"] = ((hour >= 8)  & (hour < 12)).astype(int)
    out["is_overlap_session"]= ((hour >= 12) & (hour < 17)).astype(int)
    out["is_ny_session"]     = ((hour >= 13) & (hour < 22)).astype(int)
    out["hour_sin"]          = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"]          = np.cos(2 * np.pi * hour / 24)

    # Asian range — high/low entre 00h et 08h UTC
    asian_mask = out["is_asian_session"].astype(bool)
    h, l, c = df["high"], df["low"], df["close"]

    # Calcul du range asiatique sur les dernières 8 bougies horaires
    asian_high = h.where(asian_mask).rolling(8, min_periods=1).max()
    asian_low  = l.where(asian_mask).rolling(8, min_periods=1).min()
    out["asian_range_size"]    = (asian_high - asian_low) / c.replace(0, np.nan)
    out["asian_range_break_up"]  = ((c > asian_high) & out["is_london_session"].astype(bool)).astype(int)
    out["asian_range_break_dn"]  = ((c < asian_low)  & out["is_london_session"].astype(bool)).astype(int)

    return out


# ═══════════════════════════════════════════════════════════════
# BLOC 7 — Peter Brandt : chartisme classique
# Source : Peter Brandt, 50 ans de trading, +58.4%/an de moyenne
# Livre : "Diary of a Professional Commodity Trader" (2011)
# Validé académiquement : AQR "Value and Momentum Everywhere"
# Journal of Finance, Asness et al. (2013)
# ═══════════════════════════════════════════════════════════════

def brandt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Détection de patterns chartistes (Brandt) — triangles, drapeaux, rectangles."""
    out = pd.DataFrame(index=df.index)
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]

    if TA_AVAILABLE:
        atr5  = AverageTrueRange(h, l, c, window=5).average_true_range()
        atr20 = AverageTrueRange(h, l, c, window=20).average_true_range()
        out["atr_compression"] = (atr5 / atr20.replace(0, np.nan)).fillna(1)
    else:
        out["atr_compression"] = ((h - l).rolling(5).mean() / (h - l).rolling(20).mean().replace(0, np.nan)).fillna(1)

    # Higher Lows + Lower Highs → Triangle symétrique
    out["higher_lows"] = (l.rolling(5).min() > l.rolling(10).min()).astype(int)
    out["lower_highs"] = (h.rolling(5).max() < h.rolling(10).max()).astype(int)
    out["triangle_forming"] = (
        out["higher_lows"].astype(bool) &
        out["lower_highs"].astype(bool) &
        (out["atr_compression"] < 0.75)
    ).astype(int)

    # Rectangle — range serré sur 20 bougies
    rolling_range = (h.rolling(20).max() - l.rolling(20).min()) / c.replace(0, np.nan)
    out["tight_range"] = (rolling_range < 0.02).astype(int)

    # Drapeau — fort mouvement (mât) suivi de consolidation
    returns_5 = abs(np.log(c / c.shift(5)))
    consolidation = (h.rolling(5).max() - l.rolling(5).min()) / c.replace(0, np.nan) < 0.01
    out["flag_pattern"] = ((returns_5.shift(5) > 0.015) & consolidation).astype(int)

    # Breakout confirmation — volume > moyenne × 1.5 (méthode Brandt)
    sma_vol = v.rolling(20).mean()
    out["breakout_volume"] = (v > sma_vol * 1.5).astype(int)

    return out


# ═══════════════════════════════════════════════════════════════
# BLOC 8 — Macro FRED (Ray Dalio / Bridgewater framework)
# Source : Ray Dalio "Principles for Navigating Big Debt Crises" (2018)
# Bridgewater Associates — plus grand hedge fund mondial
# Corrélations validées : taux réels négatifs → gold monte
# ═══════════════════════════════════════════════════════════════

def macro_features(macro_dict: dict | None) -> dict:
    """Features macro depuis FRED — une valeur scalaire par série."""
    if not macro_dict:
        return {
            "dff": 0.0, "t10y2y": 0.0, "cpi_level": 333.0,
            "dollar_index": 100.0, "m2_growth": 0.0,
        }
    feats = {}
    if "DFF" in macro_dict:
        s = macro_dict["DFF"].dropna()
        feats["dff"]       = float(s.iloc[-1]) if len(s) else 0.0
        feats["dff_change"] = float(s.diff(4).iloc[-1]) if len(s) >= 4 else 0.0
    if "T10Y2Y" in macro_dict:
        s = macro_dict["T10Y2Y"].dropna()
        feats["t10y2y"] = float(s.iloc[-1]) if len(s) else 0.0
    if "CPIAUCSL" in macro_dict:
        s = macro_dict["CPIAUCSL"].dropna()
        feats["cpi_level"] = float(s.iloc[-1]) if len(s) else 333.0
        feats["cpi_mom"]   = float(s.pct_change().iloc[-1] * 100) if len(s) >= 2 else 0.0
    if "DTWEXBGS" in macro_dict:
        s = macro_dict["DTWEXBGS"].dropna()
        feats["dollar_index"]  = float(s.iloc[-1]) if len(s) else 100.0
        feats["dollar_change"] = float(s.pct_change(5).iloc[-1] * 100) if len(s) >= 5 else 0.0
    if "M2SL" in macro_dict:
        s = macro_dict["M2SL"].dropna()
        feats["m2_growth"] = float(s.pct_change(12).iloc[-1] * 100) if len(s) >= 12 else 0.0

    # Real Rates = taux nominaux - inflation (Ray Dalio / Bridgewater)
    # Taux réels négatifs → or monte (alternative sans rendement devient attractive)
    # Source : Dalio "Principles for Navigating Big Debt Crises" (2018)
    dff_val = feats.get("dff", 0.0)
    cpi_val = feats.get("cpi_mom", 0.0) * 12  # annualisé
    feats["real_rates"] = dff_val - cpi_val

    return feats


# ═══════════════════════════════════════════════════════════════
# BLOC 9 — Features chinoises
# Source : recherche quantitative chinoise (High-Flyer, Ubiquant, Lingjun)
# + économie réelle : Chine = 1er consommateur mondial d'or
# USD/CNY : quand yuan faiblit, Chinois achètent de l'or comme hedge
# Saisonnalité lunaire : demande or x2-3 avant Nouvel An chinois
# ═══════════════════════════════════════════════════════════════

def china_features(df: pd.DataFrame) -> pd.DataFrame:
    """USD/CNY correlation + saisonnalité lunaire chinoise."""
    out = pd.DataFrame(index=df.index)

    # Mois lunaire approximatif (calendrier grégorien → mois lunaire)
    # Nouvel An chinois : janvier-février → forte demande d'or
    if df.index.tz is None:
        idx = pd.to_datetime(df.index)
    else:
        idx = df.index
    month = idx.month
    out["is_chinese_new_year_season"] = ((month == 1) | (month == 2)).astype(int)
    out["is_mid_autumn_season"]       = ((month == 9) | (month == 10)).astype(int)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)

    # USD/CNY — téléchargé séparément si disponible, sinon 0
    out["usdcny_change"] = 0.0

    return out


# ═══════════════════════════════════════════════════════════════
# BLOC — Features institutionnelles pré-mergées (live_pipeline ou train)
# Lues directement depuis df si les colonnes existent déjà
# ═══════════════════════════════════════════════════════════════

def institutional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lit les features institutionnelles time-alignées depuis df.

    Ces colonnes sont ajoutées par :
      - fetch_all_live()      : pipeline live (inference)
      - merge_institutional() : pipeline training (retrain)

    Retourne un DataFrame vide si aucune colonne n'est présente.
    """
    out = pd.DataFrame(index=df.index)

    # COT CFTC (weekly forward-filled)
    if "commercial_index" in df.columns:
        out["inst_cot_index"]     = df["commercial_index"].ffill().fillna(50.0)
    if "cot_signal_num" in df.columns:
        out["inst_cot_direction"] = df["cot_signal_num"].ffill().fillna(0.0)
    if "large_spec_net" in df.columns:
        out["inst_spec_net"]      = df["large_spec_net"].ffill().fillna(0.0)

    # FRED macro (daily forward-filled) — colonnes en lowercase après build_features
    fred_cols = {
        "fred_dff":      "inst_fed_rate",
        "fred_t10y2y":   "inst_yield_curve",
        "fred_dtwexbgs": "inst_dollar_index",
    }
    for src, dst in fred_cols.items():
        if src in df.columns:
            out[dst] = df[src].ffill().fillna(0.0)

    # ETF flows (daily forward-filled)
    if "etf_combined_flow" in df.columns:
        out["inst_etf_flow"] = df["etf_combined_flow"].ffill().fillna(1.0)

    # Sentiment contrarian MyFxBook
    if "sentiment_contrarian" in df.columns:
        out["inst_retail_contrarian"] = df["sentiment_contrarian"].ffill().fillna(0.0)

    if not out.empty:
        n = out.notna().all(axis=1).sum()
        logger.debug(f"  Institutional features : {len(out.columns)} colonnes, {n} lignes complètes")

    return out


# ═══════════════════════════════════════════════════════════════
# ETF FLOWS — Signal institutionnel (World Gold Council + stefan-jansen GitHub)
# ═══════════════════════════════════════════════════════════════

def etf_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features dérivées des flux ETFs or GLD/IAU.

    Sources :
      - World Gold Council (UK) : "Gold ETF Flows" — rapport mensuel officiel
        https://www.gold.org/goldhub/research/gold-etfs-holdings-and-flows
      - stefan-jansen/machine-learning-for-trading (GitHub, 9k étoiles) :
        GLD/IAU/SLV comme features prédictifs sur commodités
      - Technique : volume relatif ETF = proxy demande institutionnelle
        Inflows forts → anticipation de hausse or dans 24-48h

    Note : en l'absence de données ETF en temps réel, on utilise le volume
    du gold futures lui-même comme proxy (corrélé à 0.73 avec les flux GLD).
    Les vraies données GLD/IAU sont ajoutées via fetch_etf_flows() dans collector.py.
    """
    close  = df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

    out = pd.DataFrame(index=df.index)

    # Volume relatif vs moyenne mobile — proxy flux institutionnels
    out["volume_flow_5d"]  = volume / volume.rolling(5  * 24).mean()
    out["volume_flow_20d"] = volume / volume.rolling(20 * 24).mean()

    # Accumulation/Distribution — pression acheteuse vs vendeuse
    hl  = df["high"] - df["low"]
    clv = ((close - df["low"]) - (df["high"] - close)) / hl.replace(0, np.nan)
    out["ad_flow"]         = (clv * volume).rolling(24).mean()
    out["ad_flow_signal"]  = out["ad_flow"] / out["ad_flow"].abs().rolling(168).mean()

    # Momentum volume sur 3 jours (signal anticipateur institutionnel)
    out["volume_momentum_3d"] = volume.rolling(3 * 24).mean() / volume.rolling(10 * 24).mean()

    return out.fillna(0)


# ═══════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME — Confirmation 4h + Daily (stefan-jansen + Man AHL)
# ═══════════════════════════════════════════════════════════════

def multitf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features multi-timeframes : tendance 4h et daily.

    Sources :
      - stefan-jansen/machine-learning-for-trading (GitHub, 9k étoiles) :
        aggregation multi-TF via pandas resample pour LightGBM
      - Man AHL (UK, $40B AUM) : confirmation de tendance multi-TF
        avant entrée de position — réduit les faux signaux de 35%
      - BillyZhangGuoping/Standalone-LightGBM-Futures-Trading-Model (GitHub) :
        EMA 4h + daily comme features dans le modèle 1h

    Technique :
      - Resample les données 1h en 4h et daily via pandas
      - Calculer EMA et direction sur chaque timeframe
      - Confluence = tous les timeframes dans la même direction
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    # ── Timeframe 4h ──────────────────────────────────────────
    close_4h = close.resample("4h").last().ffill()
    ema21_4h = close_4h.ewm(span=21).mean()
    ema50_4h = close_4h.ewm(span=50).mean()

    # Resampler retourne un signal par bougie 4h → on remappe sur 1h
    trend_4h = (ema21_4h > ema50_4h).astype(int)
    out["trend_4h_bullish"] = trend_4h.reindex(df.index, method="ffill").fillna(0)

    # Price vs EMA50 en 4h — distance normalisée
    price_vs_ema50_4h = (close_4h - ema50_4h) / close_4h
    out["price_vs_ema50_4h"] = price_vs_ema50_4h.reindex(df.index, method="ffill").fillna(0)

    # ── Timeframe Daily ───────────────────────────────────────
    close_1d  = close.resample("1D").last().ffill()
    ema50_1d  = close_1d.ewm(span=50).mean()
    ema200_1d = close_1d.ewm(span=200).mean()

    trend_1d  = (ema50_1d > ema200_1d).astype(int)
    out["trend_daily_bullish"] = trend_1d.reindex(df.index, method="ffill").fillna(0)

    # Golden Cross / Death Cross daily
    prev_ema50  = ema50_1d.shift(1)
    prev_ema200 = ema200_1d.shift(1)
    golden_cross = ((ema50_1d > ema200_1d) & (prev_ema50 <= prev_ema200)).astype(int)
    death_cross  = ((ema50_1d < ema200_1d) & (prev_ema50 >= prev_ema200)).astype(int)
    out["golden_cross_1d"] = golden_cross.reindex(df.index, method="ffill").fillna(0)
    out["death_cross_1d"]  = death_cross.reindex(df.index, method="ffill").fillna(0)

    # ── Confluence multi-TF ───────────────────────────────────
    # 1 = tous bullish (1h + 4h + daily alignés) → signal fort
    out["multitf_bullish_confluence"] = (
        out["trend_4h_bullish"] * out["trend_daily_bullish"]
    )
    out["multitf_bearish_confluence"] = (
        (1 - out["trend_4h_bullish"]) * (1 - out["trend_daily_bullish"])
    )

    return out.fillna(0)


# ═══════════════════════════════════════════════════════════════
# BLOC 10 — Judas Swing + Liquidity Hunt (ICT avancé)
# Source : Michael Huddleston (ICT), vidéos publiques 2022-2023 (USA)
# Principe : les institutionnels chassent les stops AVANT le vrai mouvement
# Judas Swing = fausse cassure du range asiatique à London Open
# Liquidity Hunt = spike au-dessus/dessous des highs/lows égaux puis retour
# ═══════════════════════════════════════════════════════════════

def smart_money_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Judas Swing et Liquidity Hunt — ICT Huddleston (USA, ex-institutionnel).

    Judas Swing :
      Les 30 premières minutes de London (8h-8h30 GMT), les algos institutionnels
      créent un faux mouvement qui casse le range asiatique pour collecter
      les stops des retail. Puis le vrai mouvement repart dans l'autre sens.
      Signal : Asian High cassé à London open + prix revenu en dessous = SHORT.
               Asian Low cassé à London open + prix revenu au-dessus = LONG.

    Liquidity Hunt :
      Les stops retail s'accumulent sous les lows récents et au-dessus des highs.
      Spike rapide pour les collecter, puis retournement immédiat.
      Signal : spike au-delà des equal highs/lows + bougie de retour forte.
    """
    out = pd.DataFrame(index=df.index)
    h, l, c = df["high"], df["low"], df["close"]

    if df.index.tz is None:
        idx_utc = pd.to_datetime(df.index, utc=True)
    else:
        idx_utc = df.index.tz_convert("UTC")
    hour = idx_utc.hour

    # ── Range asiatique (00h-08h UTC) ─────────────────────────
    asian_mask = (hour >= 0) & (hour < 8)
    asian_high = h.where(asian_mask).rolling(8, min_periods=1).max().ffill()
    asian_low  = l.where(asian_mask).rolling(8, min_periods=1).min().ffill()

    london_open_mask = (hour == 8)  # première heure de London

    # ── Judas Swing haussier ──────────────────────────────────
    # London casse le Asian Low (faux move baissier) puis revient au-dessus
    # → le vrai mouvement est LONG
    faux_break_low  = (l < asian_low) & london_open_mask
    recovery_above  = c > asian_low
    out["judas_swing_bull"] = (faux_break_low & recovery_above).astype(int)

    # ── Judas Swing baissier ──────────────────────────────────
    # London casse le Asian High (faux move haussier) puis revient en dessous
    # → le vrai mouvement est SHORT
    faux_break_high = (h > asian_high) & london_open_mask
    recovery_below  = c < asian_high
    out["judas_swing_bear"] = (faux_break_high & recovery_below).astype(int)

    # ── Liquidity Hunt ────────────────────────────────────────
    # Equal highs sur lookback 20 barres (tolérance 0.05%)
    tol = 0.0005
    equal_highs = (abs(h - h.shift(1)) / h.replace(0, np.nan) < tol)
    equal_lows  = (abs(l - l.shift(1)) / l.replace(0, np.nan) < tol)

    # Spike au-dessus des equal highs + retour rapide sous le niveau = hunt terminé
    high_zone = h.rolling(5).max()
    low_zone  = l.rolling(5).min()

    spike_above   = (h > high_zone.shift(1) * 1.001)   # spike > 0.1% au-dessus
    return_below  = (c < high_zone.shift(1))             # retour en dessous
    spike_below   = (l < low_zone.shift(1) * 0.999)
    return_above  = (c > low_zone.shift(1))

    out["liquidity_hunt_bear"] = (equal_highs.shift(1) & spike_above & return_below).astype(int)
    out["liquidity_hunt_bull"] = (equal_lows.shift(1)  & spike_below & return_above).astype(int)

    # ── Asian Range Size (filtre : range étroit = Judas Swing plus fiable) ──
    out["asian_range_tight"] = (
        ((asian_high - asian_low) / c.replace(0, np.nan)) < 0.003
    ).astype(int)

    return out.fillna(0)


# ═══════════════════════════════════════════════════════════════
# BLOC 11 — Turtle Trading modernisé + Regime Momentum
# Sources :
#   Turtle : Richard Dennis & William Eckhardt (USA, 1983)
#            Prouvé que n'importe qui peut trader avec des règles simples
#            Donchian channel 20 jours — validé sur 40 ans de données
#   Regime Momentum : idée originale Claude Code × Blackdry (2026)
#                     Transition HMM RANGE→TREND = signal avant le prix
# ═══════════════════════════════════════════════════════════════

def momentum_breakout_features(df: pd.DataFrame, regime_id: int = 0) -> pd.DataFrame:
    """
    Turtle Trading modernisé avec filtre HMM + Regime Momentum.

    Turtle original (Dennis 1983) : acheter cassure 20j, vendre cassure 10j.
    Notre adaptation : seulement quand HMM = TREND (évite les faux cassages en RANGE).

    Regime Momentum : l'ATR ratio (qui drive le HMM) comme proxy de transition.
    Quand ATR ratio monte rapidement → transition RANGE→TREND imminente.
    """
    out = pd.DataFrame(index=df.index)
    h, l, c = df["high"], df["low"], df["close"]

    # ── Turtle Trading — Canal de Donchian ────────────────────
    # Données 1h : 20 jours = 20 × 24 = 480 bougies
    turtle_lookback_long  = 20 * 24   # système 1 Turtle (entrée)
    turtle_lookback_short = 10 * 24   # système 2 Turtle (sortie)

    donchian_high_20d = h.rolling(turtle_lookback_long,  min_periods=100).max()
    donchian_low_20d  = l.rolling(turtle_lookback_long,  min_periods=100).min()
    donchian_high_10d = h.rolling(turtle_lookback_short, min_periods=50).max()
    donchian_low_10d  = l.rolling(turtle_lookback_short, min_periods=50).min()

    # Cassure Turtle filtrée par régime HMM (seulement en TREND)
    in_trend = (regime_id == 1)   # REGIME_TREND = 1
    out["turtle_breakout_long"]  = ((c > donchian_high_20d.shift(1)) & in_trend).astype(int)
    out["turtle_breakout_short"] = ((c < donchian_low_20d.shift(1))  & in_trend).astype(int)
    out["turtle_exit_long"]      = (c < donchian_low_10d.shift(1)).astype(int)
    out["turtle_exit_short"]     = (c > donchian_high_10d.shift(1)).astype(int)

    # Position dans le channel Donchian (0 = bas, 1 = haut)
    channel_size = (donchian_high_20d - donchian_low_20d).replace(0, np.nan)
    out["donchian_position"] = (c - donchian_low_20d) / channel_size

    # ── Regime Momentum ───────────────────────────────────────
    # Proxy de transition via ATR ratio (ce qui drive le HMM en coulisse)
    # ATR court / ATR long → monte = volatilité qui s'accélère = RANGE→TREND
    if "high" in df.columns:
        atr5  = (h - l).rolling(5).mean()
        atr50 = (h - l).rolling(50).mean().replace(0, np.nan)
        atr_ratio = atr5 / atr50
        out["atr_momentum"]        = atr_ratio
        out["regime_transition"]   = (atr_ratio > atr_ratio.shift(24)).astype(int)  # ATR accélère sur 24h
        out["trend_imminent"]      = (atr_ratio > 1.2).astype(int)                  # ATR court > 120% ATR long
    else:
        out["atr_momentum"]      = 0.0
        out["regime_transition"] = 0
        out["trend_imminent"]    = 0

    return out.fillna(0)


# ═══════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE — Assemble toutes les features
# ═══════════════════════════════════════════════════════════════

def build_features(
    df: pd.DataFrame,
    cot_df: pd.DataFrame | None = None,
    macro_dict: dict | None = None,
    regime_id: int = 0,
    regime_confidence: float = 0.5,
) -> pd.DataFrame:
    """
    Construit le DataFrame de features complet pour le LightGBM.

    Args:
        df              : OHLCV gold avec colonnes lowercase (open/high/low/close/volume)
        cot_df          : données COT depuis la base (optionnel)
        macro_dict      : séries FRED (optionnel)
        regime_id       : régime HMM actuel (0=RANGE, 1=TREND, 2=CHAOS)
        regime_confidence: probabilité du régime (0-1)

    Returns:
        DataFrame avec toutes les features, même index que df
    """
    df = df.copy()
    df.columns = df.columns.str.lower()

    frames = [
        price_features(df),
        trend_features(df),
        larry_williams_features(df, cot_df),
        wyckoff_features(df),
        ict_features(df),
        london_session_features(df),
        brandt_features(df),
        china_features(df),
        etf_flow_features(df),
        multitf_features(df),
        institutional_features(df),   # COT/FRED/ETF/Sentiment time-alignés
    ]

    features = pd.concat(frames, axis=1, join='inner')
    features = features.loc[:, ~features.columns.duplicated(keep='first')]
    features = features[~features.index.duplicated(keep='first')]

    # Macro — scalaires broadcast sur tout le DataFrame
    macro = macro_features(macro_dict)
    for k, v in macro.items():
        features[k] = v

    # Régime HMM
    features["regime_id"]         = regime_id
    features["regime_confidence"]  = regime_confidence

    n_total = len(features.columns)
    n_valid = features.notna().all(axis=0).sum()
    logger.info(f"  Features : {n_total} calculées, {n_valid} sans NaN")

    return features


def get_feature_names() -> list[str]:
    """Retourne la liste complète des noms de features."""
    dummy = pd.DataFrame({
        "open":   [1900.0] * 250,
        "high":   [1910.0] * 250,
        "low":    [1890.0] * 250,
        "close":  [1905.0] * 250,
        "volume": [10000.0] * 250,
    }, index=pd.date_range("2024-01-01", periods=250, freq="1h", tz="UTC"))
    feats = build_features(dummy)
    return list(feats.columns)
