"""
Live Pipeline — Agrège toutes les sources de données institutionnelles en temps réel.

Sources :
  - OHLCV 1h     : yfinance (GC=F)                    — toujours frais
  - COT CFTC     : CFTC.gov (positions institutionnelles) — cache 7 jours
  - FRED Macro   : FRED API (taux réels, DXY, inflation) — cache 24h
  - ETF Flows    : yfinance (GLD, IAU, GDX)             — cache 4h
  - Sentiment    : MyFxBook Community Outlook            — cache 1h

Usage :
    from src.data.live_pipeline import fetch_all_live
    df = fetch_all_live()   # DataFrame 1h enrichi, prêt pour feature_builder
"""

import os
import time
import requests
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Cache en mémoire (évite de re-fetcher COT/FRED toutes les 5 min) ─────────
_CACHE: dict[str, tuple] = {}


def _cache_get(key: str):
    if key in _CACHE:
        data, expires = _CACHE[key]
        if time.time() < expires:
            return data
    return None


def _cache_set(key: str, data, ttl_seconds: int):
    _CACHE[key] = (data, time.time() + ttl_seconds)


# ── 1. OHLCV ─────────────────────────────────────────────────────────────────

def fetch_ohlcv(period: str = "120d") -> pd.DataFrame:
    """Bougies 1h gold depuis yfinance. Pas de cache — on veut toujours le dernier prix."""
    import yfinance as yf
    try:
        df = yf.download("GC=F", period=period, interval="1h",
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if hasattr(df.columns, "get_level_values"):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index, utc=True)
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as e:
        logger.error(f"❌ OHLCV : {e}")
        return pd.DataFrame()


# ── 2. COT CFTC (cache 7 jours) ──────────────────────────────────────────────

def fetch_cot_latest() -> dict:
    """
    Dernières valeurs COT gold (positions commerciales/spéculatives).
    Publication hebdomadaire CFTC — inutile de refetcher plus souvent.
    """
    cached = _cache_get("cot")
    if cached is not None:
        logger.debug("  COT : depuis cache")
        return cached

    from src.data.collector import fetch_cot_gold
    df = fetch_cot_gold()

    if df.empty:
        result = {
            "commercial_index": 50.0,
            "cot_signal_num": 0.0,
            "large_spec_net": 0.0,
        }
    else:
        last = df.iloc[-1]
        signal = str(last.get("cot_signal", "NEUTRAL"))
        result = {
            "commercial_index": float(last.get("commercial_index", 50.0)),
            "large_spec_net":   float(last.get("large_spec_net", 0.0)),
            "cot_signal_num":   1.0 if signal == "BULLISH" else (-1.0 if signal == "BEARISH" else 0.0),
        }
        logger.info(f"  COT : index={result['commercial_index']:.1f} | signal={signal}")

    _cache_set("cot", result, ttl_seconds=7 * 24 * 3600)
    return result


# ── 3. FRED Macro (cache 24h) ─────────────────────────────────────────────────

def fetch_fred_latest() -> dict:
    """
    Dernières valeurs macro FRED.
    Taux réels (DFII10), DXY (DTWEXBGS), inflation (T10YIE).
    """
    cached = _cache_get("fred")
    if cached is not None:
        logger.debug("  FRED : depuis cache")
        return cached

    from src.data.collector import fetch_fred_macro
    raw = fetch_fred_macro()

    result = {}
    for series_id, series in raw.items():
        if hasattr(series, "dropna") and not series.empty:
            result[f"fred_{series_id}"] = float(series.dropna().iloc[-1])

    # Real rate signal : taux réels négatifs = or favorable
    if "fred_DFII10" in result:
        result["real_rate_signal"] = -1.0 if result["fred_DFII10"] < 0 else 1.0

    _cache_set("fred", result, ttl_seconds=24 * 3600)
    logger.info(f"  FRED : {len(result)} séries macro")
    return result


# ── 4. ETF Flows (cache 4h) ───────────────────────────────────────────────────

def fetch_etf_latest() -> dict:
    """
    Flux institutionnels via GLD et IAU.
    Volume relatif vs moyenne 20j = proxy d'entrées/sorties institutionnelles.
    """
    cached = _cache_get("etf")
    if cached is not None:
        logger.debug("  ETF : depuis cache")
        return cached

    from src.data.collector import fetch_etf_flows
    df = fetch_etf_flows(days=60)

    if df.empty:
        result = {"etf_combined_flow": 1.0, "gld_flow_signal": 1.0, "iau_flow_signal": 1.0}
    else:
        last = df.iloc[-1]
        result = {
            "etf_combined_flow": float(last.get("etf_combined_flow", 1.0)),
            "gld_flow_signal":   float(last.get("GLD_flow_signal", 1.0)),
            "iau_flow_signal":   float(last.get("IAU_flow_signal", 1.0)),
        }
        logger.info(
            f"  ETF flows : GLD={result['gld_flow_signal']:.2f} "
            f"IAU={result['iau_flow_signal']:.2f} "
            f"combiné={result['etf_combined_flow']:.2f}"
        )

    _cache_set("etf", result, ttl_seconds=4 * 3600)
    return result


# ── 5. MyFxBook Sentiment (cache 1h) ─────────────────────────────────────────

def fetch_myfxbook_sentiment() -> dict:
    """
    Positionnement retail gold depuis MyFxBook Community Outlook.
    Signal contrarian : >70% longs → SHORT | <30% longs → LONG.
    Remplace OANDA (non disponible en Afrique).
    """
    cached = _cache_get("myfxbook")
    if cached is not None:
        logger.debug("  MyFxBook : depuis cache")
        return cached

    default = {"retail_long_pct": 0.5, "retail_short_pct": 0.5, "sentiment_contrarian": 0.0}
    email    = os.getenv("MYFXBOOK_EMAIL", "")
    password = os.getenv("MYFXBOOK_PASSWORD", "")

    if not email or not password:
        _cache_set("myfxbook", default, 3600)
        return default

    try:
        # Login
        r = requests.get(
            "https://www.myfxbook.com/api/login.json",
            params={"email": email, "password": password},
            timeout=10,
        )
        data = r.json()
        if data.get("error") or not data.get("session"):
            logger.warning("⚠️ MyFxBook login échoué")
            _cache_set("myfxbook", default, 3600)
            return default

        from urllib.parse import unquote
        session = unquote(data["session"])

        # Community Outlook
        r2 = requests.get(
            "https://www.myfxbook.com/api/get-community-outlook.json",
            params={"session": session},
            timeout=10,
        )
        outlook = r2.json()
        if outlook.get("error"):
            _cache_set("myfxbook", default, 3600)
            return default

        symbols_raw = outlook.get("symbols", [])
        if isinstance(symbols_raw, dict):
            symbols = symbols_raw.get("symbol", [])
        else:
            symbols = symbols_raw
        gold = next(
            (s for s in symbols if "XAU" in s.get("name", "").upper()
             or "GOLD" in s.get("name", "").upper()),
            None,
        )

        if gold:
            long_pct  = float(gold.get("longPercentage", 50)) / 100
            short_pct = 1.0 - long_pct
            if long_pct > 0.70:
                contrarian = -1.0   # Crowd trop long → on vend
            elif long_pct < 0.30:
                contrarian = 1.0    # Crowd trop short → on achète
            else:
                contrarian = round((0.5 - long_pct) * 2, 2)

            result = {
                "retail_long_pct":     long_pct,
                "retail_short_pct":    short_pct,
                "sentiment_contrarian": contrarian,
            }
            logger.info(
                f"  MyFxBook : {long_pct:.0%} longs | contrarian={contrarian:+.2f}"
            )
        else:
            result = default

        _cache_set("myfxbook", result, 3600)
        return result

    except Exception as e:
        logger.warning(f"⚠️ MyFxBook sentiment : {e}")
        _cache_set("myfxbook", default, 3600)
        return default


# ── 6. NIM Sentiment (cache 1h via sentiment_analyzer) ───────────────────────

def fetch_nim_sentiment() -> dict:
    """Score de sentiment news gold via DeepSeek v4 Flash (NVIDIA NIM). Cache 1h."""
    cached = _cache_get("nim_sentiment")
    if cached is not None:
        logger.debug("  NIM sentiment : depuis cache")
        return cached

    default = {"nim_sentiment": 0.0}
    try:
        from src.ai.sentiment_analyzer import get_sentiment_score
        result_nim = get_sentiment_score()
        result = {"nim_sentiment": result_nim.get("nim_sentiment", 0.0)}
        logger.info(
            f"  NIM sentiment : {result['nim_sentiment']:+.2f} "
            f"({result_nim.get('headlines_count', 0)} headlines)"
        )
    except Exception as e:
        logger.warning(f"⚠️ NIM sentiment : {e}")
        result = default

    _cache_set("nim_sentiment", result, 3600)
    return result


# ── Pipeline Principal ────────────────────────────────────────────────────────

def fetch_all_live(period: str = "120d") -> pd.DataFrame | None:
    """
    Point d'entrée unique du pipeline.

    Télécharge OHLCV + données institutionnelles et retourne un DataFrame
    1h enrichi, prêt à passer dans feature_builder / SignalGenerator.

    Les données basse fréquence (COT hebdo, FRED mensuel) sont forward-fillées
    sur toutes les bougies 1h via les valeurs scalaires en cache.

    Returns:
        DataFrame enrichi ou None si OHLCV indisponible.
    """
    logger.debug("🔄 Pipeline live — collecte en cours...")

    df = fetch_ohlcv(period=period)
    if df.empty or len(df) < 200:
        logger.error("❌ OHLCV insuffisant pour générer un signal")
        return None

    # Ajout des colonnes institutionnelles (scalaires forward-fillées sur tout le df)
    for key, val in fetch_cot_latest().items():
        if isinstance(val, (int, float)):
            df[key] = val

    for key, val in fetch_fred_latest().items():
        if isinstance(val, (int, float)):
            df[key] = val

    for key, val in fetch_etf_latest().items():
        if isinstance(val, (int, float)):
            df[key] = val

    for key, val in fetch_myfxbook_sentiment().items():
        if isinstance(val, (int, float)):
            df[key] = val

    for key, val in fetch_nim_sentiment().items():
        if isinstance(val, (int, float)):
            df[key] = val

    logger.debug(f"✅ Pipeline : {len(df)} bougies | {len(df.columns)} colonnes")
    return df


# ── Test standalone ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from loguru import logger
    logger.remove()
    logger.add(sys.stdout, level="DEBUG", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    logger.info("Test du pipeline live...")
    df = fetch_all_live()

    if df is not None:
        logger.success(f"✅ DataFrame : {len(df)} lignes × {len(df.columns)} colonnes")
        logger.info(f"  Dernière bougie : {df.index[-1]}")
        logger.info(f"  Prix gold actuel : {df['close'].iloc[-1]:.2f}$")

        extra_cols = [c for c in df.columns if c not in ["open","high","low","close","volume"]]
        logger.info(f"  Colonnes institutionnelles ({len(extra_cols)}) : {extra_cols}")
    else:
        logger.error("❌ Pipeline échoué")
