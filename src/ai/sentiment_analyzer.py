"""
Sentiment Analyzer — Analyse des news gold via DeepSeek (NVIDIA NIM local).

Fonctionnement :
  1. Récupère les dernières headlines gold (yfinance + FXStreet RSS)
  2. Envoie au modèle DeepSeek via NIM (OpenAI-compatible API)
  3. Retourne un score de sentiment -1.0 à +1.0

Cache 1h — les news changent moins vite que le prix.

Variables .env requises :
  NIM_BASE_URL = http://localhost:8000/v1
  NIM_API_KEY  = nim-local-key
"""

import os
import time
import json
import requests
import yfinance as yf
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

NIM_BASE_URL = os.getenv("NIM_BASE_URL", "http://localhost:8000/v1")
NIM_API_KEY  = os.getenv("NIM_API_KEY", "nim-local-key")
NIM_MODEL    = "deepseek-ai/deepseek-r1"  # adapter selon le modèle chargé dans NIM

_CACHE: dict[str, tuple] = {}


def _cache_get(key: str):
    if key in _CACHE:
        data, expires = _CACHE[key]
        if time.time() < expires:
            return data
    return None


def _cache_set(key: str, data, ttl_seconds: int):
    _CACHE[key] = (data, time.time() + ttl_seconds)


# ── Collecte des headlines ────────────────────────────────────────────────────

def fetch_gold_headlines(max_headlines: int = 10) -> list[str]:
    """
    Récupère les dernières headlines gold depuis yfinance.
    Gratuit, pas de clé API requise.
    """
    try:
        ticker = yf.Ticker("GC=F")
        news = ticker.news or []
        headlines = []
        for item in news[:max_headlines]:
            title = item.get("title", "")
            if title:
                headlines.append(title)
        logger.debug(f"  {len(headlines)} headlines gold récupérées")
        return headlines
    except Exception as e:
        logger.warning(f"⚠️ Headlines yfinance : {e}")
        return []


def fetch_fxstreet_headlines() -> list[str]:
    """
    Headlines gold depuis FXStreet RSS (gratuit, pas d'auth).
    Source de qualité — couvre XAU/USD en temps réel.
    """
    try:
        import xml.etree.ElementTree as ET
        url = "https://rss.fxstreet.com/news/currencies/xau/usd"
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.text)
        headlines = []
        for item in root.iter("item"):
            title = item.find("title")
            if title is not None and title.text:
                headlines.append(title.text.strip())
        logger.debug(f"  {len(headlines)} headlines FXStreet")
        return headlines[:10]
    except Exception as e:
        logger.warning(f"⚠️ FXStreet RSS : {e}")
        return []


# ── Analyse DeepSeek via NIM ──────────────────────────────────────────────────

def analyze_with_nim(headlines: list[str]) -> float:
    """
    Envoie les headlines à DeepSeek (via NIM local) et retourne un score -1 à +1.

    -1.0 = très bearish or
     0.0 = neutre
    +1.0 = très bullish or
    """
    if not headlines:
        return 0.0

    headlines_text = "\n".join(f"- {h}" for h in headlines[:10])

    prompt = f"""Analyse les headlines financières suivantes sur l'or (XAU/USD).
Retourne UNIQUEMENT un score JSON entre -1.0 (très bearish) et +1.0 (très bullish).
Ne rien ajouter d'autre que le JSON.

Headlines :
{headlines_text}

Réponse attendue (exemple) :
{{"score": 0.3, "reason": "légèrement bullish — hausse des tensions géopolitiques"}}"""

    try:
        r = requests.post(
            f"{NIM_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {NIM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": NIM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.1,
            },
            timeout=15,
        )

        if r.status_code != 200:
            logger.warning(f"⚠️ NIM API erreur {r.status_code}")
            return 0.0

        content = r.json()["choices"][0]["message"]["content"].strip()

        # Parser le JSON retourné par le modèle
        # Nettoyer si le modèle ajoute du texte autour
        start = content.find("{")
        end   = content.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(content[start:end])
            score  = float(parsed.get("score", 0.0))
            reason = parsed.get("reason", "")
            score  = max(-1.0, min(1.0, score))
            logger.info(f"  DeepSeek NIM : score={score:+.2f} | {reason}")
            return score

        logger.warning(f"⚠️ NIM réponse non parseable : {content[:100]}")
        return 0.0

    except requests.exceptions.ConnectionError:
        logger.warning("⚠️ NIM non disponible (localhost:8000) — score neutre")
        return 0.0
    except Exception as e:
        logger.warning(f"⚠️ NIM analyse : {e}")
        return 0.0


# ── Interface principale ──────────────────────────────────────────────────────

def get_sentiment_score() -> dict:
    """
    Retourne le score de sentiment actuel sur les news gold.
    Cache 1h — appel rapide à chaque cycle du bot.

    Returns:
        {
            "nim_sentiment":    float  # -1 à +1
            "headlines_count":  int
            "nim_available":    bool
        }
    """
    cached = _cache_get("sentiment")
    if cached is not None:
        logger.debug("  NIM sentiment : depuis cache")
        return cached

    # Collecter les headlines des deux sources
    headlines = fetch_gold_headlines() + fetch_fxstreet_headlines()
    headlines = list(dict.fromkeys(headlines))  # dédupliquer

    nim_score = analyze_with_nim(headlines)

    result = {
        "nim_sentiment":   nim_score,
        "headlines_count": len(headlines),
        "nim_available":   nim_score != 0.0 or len(headlines) == 0,
    }

    _cache_set("sentiment", result, ttl_seconds=3600)
    return result


# ── Test standalone ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, level="DEBUG", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    logger.info("Test sentiment analyzer...")

    headlines = fetch_gold_headlines() + fetch_fxstreet_headlines()
    logger.info(f"Headlines collectées ({len(headlines)}) :")
    for h in headlines[:5]:
        logger.info(f"  → {h}")

    result = get_sentiment_score()
    logger.success(f"Score final : {result['nim_sentiment']:+.2f}")
    logger.info(f"NIM disponible : {result['nim_available']}")
