"""
Retail Sentiment — Positionnement des traders retail sur XAU/USD.

Sources institutionnelles :
  - OANDA Position Book API v3 (OANDA Group, USA/UK)
    Endpoint public : /v3/instruments/XAU_USD/positionBook
    Donne la distribution exacte des positions retail ouvertes
  - Myfxbook Community Outlook (fallback)
    API publique : get-community-outlook.json
    Agrège les données de 50,000+ comptes réels

Logique contrarian (principe fondamental hedge funds) :
  - Source : Fade-the-Crowd strategy, utilisée par Tudor Jones, Soros
  - Quand 70%+ de retail est LONG → institutionnels sont SHORT → signal SHORT
  - Quand 30%- de retail est LONG → institutionnels sont LONG → signal LONG
  - Zone 30-70% → neutre, pas de signal

Références académiques :
  - Hoffmann & Post, "What Do Investors Do When They Trade?" (2014)
    Les retail traders ont un biais systématique de "buy the dip" → contrarian
  - Barber & Odean, "Trading Is Hazardous to Your Wealth" (2000)
    Retail perd en moyenne 3.7%/an vs le marché → fade their positions
  - Grinblatt & Keloharju, "The Investment Behavior and Performance of
    Various Investor Types" (2000) — institutionnels battent le retail

Configuration (.env) :
  OANDA_API_KEY=ton_api_key_oanda
  OANDA_ACCOUNT_TYPE=practice   # practice | live
  MYFXBOOK_EMAIL=ton_email
  MYFXBOOK_PASSWORD=ton_mdp
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
OANDA_API_KEY   = os.getenv("OANDA_API_KEY", "")
OANDA_ACCT_TYPE = os.getenv("OANDA_ACCOUNT_TYPE", "practice")
MYFXBOOK_EMAIL  = os.getenv("MYFXBOOK_EMAIL", "")
MYFXBOOK_PASS   = os.getenv("MYFXBOOK_PASSWORD", "")

OANDA_BASE = {
    "practice": "https://api-fxpractice.oanda.com",
    "live":     "https://api-fxtrade.oanda.com",
}

# Seuils contrarian
LONG_EXTREME  = 70.0   # >70% retail long → signal SHORT
SHORT_EXTREME = 30.0   # <30% retail long → signal LONG
CACHE_TTL     = 3600   # Rafraîchir toutes les heures (données changent lentement)

_cache: dict = {"data": None, "ts": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — OANDA Position Book
# ─────────────────────────────────────────────────────────────────────────────

def fetch_oanda_sentiment() -> dict | None:
    """
    Récupère le positionnement retail via OANDA Position Book API v3.

    Le Position Book montre la distribution des positions ouvertes
    par niveau de prix — on agrège pour obtenir % long / % short global.

    Nécessite : OANDA_API_KEY dans .env (compte practice gratuit sur oanda.com)
    """
    if not OANDA_API_KEY:
        logger.debug("OANDA_API_KEY absent — skip OANDA source")
        return None

    base = OANDA_BASE.get(OANDA_ACCT_TYPE, OANDA_BASE["practice"])
    url  = f"{base}/v3/instruments/XAU_USD/positionBook"

    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Content-Type":  "application/json",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        buckets     = data.get("orderBook", data.get("positionBook", {})).get("buckets", [])
        if not buckets:
            return None

        total_long  = sum(float(b.get("longCountPercent", 0)) for b in buckets)
        total_short = sum(float(b.get("shortCountPercent", 0)) for b in buckets)
        total       = total_long + total_short

        if total == 0:
            return None

        long_pct  = (total_long  / total) * 100
        short_pct = (total_short / total) * 100

        logger.debug(f"OANDA Position Book : {long_pct:.1f}% LONG / {short_pct:.1f}% SHORT")
        return {
            "source":    "OANDA",
            "long_pct":  round(long_pct, 1),
            "short_pct": round(short_pct, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.warning(f"OANDA Position Book erreur : {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — Myfxbook Community Outlook
# ─────────────────────────────────────────────────────────────────────────────

_myfxbook_session: str = ""

def _myfxbook_login() -> str:
    """Authentification Myfxbook — retourne le session token."""
    global _myfxbook_session
    if _myfxbook_session:
        return _myfxbook_session

    if not MYFXBOOK_EMAIL or not MYFXBOOK_PASS:
        return ""

    try:
        url  = "https://www.myfxbook.com/api/login.json"
        resp = requests.get(url, params={
            "login":    MYFXBOOK_EMAIL,
            "password": MYFXBOOK_PASS,
        }, timeout=10)
        data = resp.json()
        if not data.get("error", True):
            _myfxbook_session = data["session"]
            logger.debug(f"Myfxbook connecté — session {_myfxbook_session[:8]}...")
            return _myfxbook_session
    except Exception as e:
        logger.warning(f"Myfxbook login erreur : {e}")

    return ""


def fetch_myfxbook_sentiment() -> dict | None:
    """
    Récupère le positionnement XAUUSD depuis Myfxbook Community Outlook.
    Agrège 50,000+ comptes de traders réels.
    """
    session = _myfxbook_login()
    if not session:
        logger.debug("Myfxbook session absente — skip")
        return None

    try:
        url  = "https://www.myfxbook.com/api/get-community-outlook.json"
        resp = requests.get(url, params={"session": session}, timeout=10)
        data = resp.json()

        if data.get("error", True):
            return None

        symbols = {s["name"]: s for s in data.get("symbols", [])}
        gold    = symbols.get("XAUUSD") or symbols.get("XAU/USD")

        if not gold:
            return None

        long_pct  = float(gold["longPercentage"])
        short_pct = float(gold["shortPercentage"])

        logger.debug(f"Myfxbook XAUUSD : {long_pct:.1f}% LONG / {short_pct:.1f}% SHORT")
        return {
            "source":    "Myfxbook",
            "long_pct":  round(long_pct, 1),
            "short_pct": round(short_pct, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.warning(f"Myfxbook outlook erreur : {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Signal contrarian
# ─────────────────────────────────────────────────────────────────────────────

def get_sentiment_signal(long_pct: float) -> dict:
    """
    Convertit le % retail long en signal contrarian.

    Logique Tudor Jones / Soros :
      >70% long  → tout le monde est long → plus personne pour acheter
                   → les institutionnels vendent → SHORT
      <30% long  → tout le monde est short → plus personne pour vendre
                   → les institutionnels achètent → LONG
      30-70%     → zone neutre → FLAT

    Args:
        long_pct : pourcentage de retail traders long (0-100)

    Returns:
        dict avec signal, force, et valeurs brutes
    """
    short_pct = 100.0 - long_pct

    if long_pct >= LONG_EXTREME:
        signal    = "SHORT"
        strength  = min((long_pct - LONG_EXTREME) / (100 - LONG_EXTREME) * 100, 100)
    elif long_pct <= SHORT_EXTREME:
        signal    = "LONG"
        strength  = min((SHORT_EXTREME - long_pct) / SHORT_EXTREME * 100, 100)
    else:
        signal   = "FLAT"
        strength = 0.0

    return {
        "signal":    signal,
        "strength":  round(strength, 1),
        "long_pct":  long_pct,
        "short_pct": short_pct,
        "extreme_long":  long_pct >= LONG_EXTREME,
        "extreme_short": long_pct <= SHORT_EXTREME,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal (avec cache)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_retail_sentiment(use_cache: bool = True) -> dict:
    """
    Retourne le sentiment retail XAUUSD depuis la meilleure source disponible.

    Hiérarchie :
      1. OANDA Position Book (si OANDA_API_KEY configuré)
      2. Myfxbook Community Outlook (si credentials configurés)
      3. Données neutres (50/50) en fallback

    Returns:
        dict avec long_pct, short_pct, signal, strength, source
    """
    global _cache

    # Cache TTL
    if use_cache and _cache["data"] and (time.time() - _cache["ts"]) < CACHE_TTL:
        return _cache["data"]

    raw = fetch_oanda_sentiment() or fetch_myfxbook_sentiment()

    if raw:
        signal_data = get_sentiment_signal(raw["long_pct"])
        result = {**raw, **signal_data}
        logger.info(
            f"📊 Retail Sentiment [{raw['source']}] : "
            f"{raw['long_pct']:.1f}% LONG | Signal : {signal_data['signal']} "
            f"(force {signal_data['strength']:.0f}%)"
        )
    else:
        logger.warning("⚠️ Retail Sentiment : aucune source disponible — retour neutre")
        result = {
            "source":        "none",
            "long_pct":      50.0,
            "short_pct":     50.0,
            "signal":        "FLAT",
            "strength":      0.0,
            "extreme_long":  False,
            "extreme_short": False,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        }

    _cache = {"data": result, "ts": time.time()}
    return result


def get_sentiment_features() -> dict:
    """
    Retourne les features de sentiment retail pour le LightGBM.
    À appeler depuis feature_builder.py — APRÈS validation via signal_validator.

    Returns:
        dict : retail_long_pct, retail_extreme_long, retail_extreme_short,
               retail_contrarian_long, retail_contrarian_short
    """
    data = fetch_retail_sentiment()
    return {
        "retail_long_pct":         data["long_pct"] / 100.0,        # normalisé 0-1
        "retail_extreme_long":     int(data["extreme_long"]),        # 1 si >70% long
        "retail_extreme_short":    int(data["extreme_short"]),       # 1 si <30% long
        "retail_contrarian_long":  int(data["signal"] == "LONG"),    # signal contrarian LONG
        "retail_contrarian_short": int(data["signal"] == "SHORT"),   # signal contrarian SHORT
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, level="DEBUG", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    logger.info("Test Retail Sentiment — XAU/USD")
    logger.info(f"  OANDA_API_KEY  : {'✅ configuré' if OANDA_API_KEY else '❌ absent (.env)'}")
    logger.info(f"  MYFXBOOK_EMAIL : {'✅ configuré' if MYFXBOOK_EMAIL else '❌ absent (.env)'}")

    data = fetch_retail_sentiment(use_cache=False)

    logger.info("")
    logger.info("═" * 50)
    logger.info(f"  Source      : {data['source']}")
    logger.info(f"  Retail LONG : {data['long_pct']:.1f}%")
    logger.info(f"  Retail SHORT: {data['short_pct']:.1f}%")
    logger.info(f"  Signal      : {data['signal']} (force {data['strength']:.0f}%)")
    logger.info("═" * 50)

    # Simulation : que se passe-t-il à différents niveaux ?
    logger.info("\n  Simulation signaux :")
    for pct in [20, 30, 40, 50, 60, 70, 80]:
        sig = get_sentiment_signal(float(pct))
        logger.info(f"  {pct}% LONG → {sig['signal']:5s} (force {sig['strength']:.0f}%)")
