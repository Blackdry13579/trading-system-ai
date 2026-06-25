"""
NEXUS STRATEGY — Stratégie propriétaire
Architecture 4 couches : Macro → Flux → Régime → Entrée
Créée par Claude Code, 24 Juin 2026
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────────────────────
#  Constantes de pondération
# ─────────────────────────────────────────────────────────────

# Couche 1 — Macro
W_REAL_RATES   = 0.25
W_DXY_MOMENTUM = 0.20
W_COT_DIVERGE  = 0.20
W_SGX_PREMIUM  = 0.15
W_PETRODOLLAR  = 0.10
W_ZAR_PROXY    = 0.10

# Couche 2 — Flux institutionnel
W_ETF_FLOWS    = 0.25
W_SMART_MONEY  = 0.35
W_DUMB_MONEY   = 0.25   # négatif → contrarian
W_YEN_SAFE     = 0.15

# Couche 3 — Régime & Timing
W_REGIME_MOMENTUM = 0.40
W_MULTITF         = 0.30
W_TURTLE          = 0.15
W_COT_TIMING      = 0.15

# Signal final
W_REGIME_FINAL = 0.40
W_FLOW_FINAL   = 0.35
W_MACRO_FINAL  = 0.25

# Seuils de décision
NEXUS_LONG_THRESHOLD  = 65.0
NEXUS_SHORT_THRESHOLD = -65.0
ALPHA_DECAY_HALFLIFE_H = 24.0   # demi-vie des signaux = 24h

# SGX premium seuil (dollars)
SGX_PREMIUM_BULLISH_THRESHOLD = 30.0

# Seuils saisonniers Inde (mois)
INDIA_FESTIVE_MONTHS = {10, 11}     # Diwali, Dhanteras
INDIA_WEDDING_MONTHS = {1, 2, 5, 6} # Mariages

# Seuil lunar (nouvelle lune ± 2 jours = achat physique Inde/Chine)
LUNAR_PERIOD_DAYS = 29.53

# Session asiatique — multiplicateurs de signal (Chine quant)
SESSION_MULTIPLIERS = {
    (0, 2):   0.5,   # consolidation post-NY
    (2, 5):   1.2,   # PBOC/BoJ/RBI actifs — AMPLIFIER
    (5, 8):   1.0,   # pre-London positioning
    (8, 17):  0.8,   # London/NY — géré par nos filtres de session
    (17, 24): 0.6,   # fin NY / fermeture
}

# Corrélation ZAR/Gold normale = -0.83 (Afrique du Sud)
ZAR_GOLD_NORMAL_CORR = -0.83
ZAR_CORR_DEVIATION_THRESHOLD = 0.30

# Seuil stress KRW (Corée du Sud, NPS signal)
KRW_STRESS_ZSCORE = 2.0

# Ratio Or/Pétrole historique (Moyen-Orient pétrodollar)
GOLD_OIL_RATIO_MEAN = 15.5

# Demi-vie alpha decay (arXiv 2512.11913 — High-Flyer Quant)
ALPHA_DECAY_HALFLIFE_DEFAULT = 12.0   # 12h (calibré or 1h)


# ─────────────────────────────────────────────────────────────
#  Dataclasses de résultat
# ─────────────────────────────────────────────────────────────

@dataclass
class NexusLayer:
    name: str
    score: float        # -100 à +100
    components: dict = field(default_factory=dict)

    def __repr__(self):
        return f"{self.name}: {self.score:+.1f}"


@dataclass
class NexusSignal:
    direction: str          # "LONG" | "SHORT" | "FLAT"
    composite: float        # score final -100 à +100
    confidence: float       # 0 à 1
    macro: NexusLayer
    flow: NexusLayer
    regime: NexusLayer
    entry_trigger: Optional[str]
    dynamic_risk_pct: float # Kelly dynamique = risque % du capital
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_valid(self) -> bool:
        return self.direction != "FLAT"

    def summary(self) -> str:
        return (
            f"NEXUS {self.direction} | composite={self.composite:+.1f} "
            f"| conf={self.confidence:.0%} | risk={self.dynamic_risk_pct:.2%}"
        )


# ─────────────────────────────────────────────────────────────
#  COUCHE 1 — MACRO BIAS
# ─────────────────────────────────────────────────────────────

def _real_rates_signal(real_rates: float) -> float:
    """
    Taux réels USA (TIPS 10Y) → score macro.
    < -1.5% = très bullish or (+100)
    > +2.0% = très bearish or (-100)
    Source : Bridgewater / Ray Dalio
    """
    if real_rates < -1.5:
        return 100.0
    elif real_rates < -1.0:
        return 80.0
    elif real_rates < -0.5:
        return 50.0
    elif real_rates < 0.0:
        return 20.0
    elif real_rates < 1.0:
        return -20.0
    elif real_rates < 2.0:
        return -60.0
    else:
        return -100.0


def _dxy_momentum_signal(dxy_returns_20d: float) -> float:
    """
    Dollar 20j momentum → signal inverse (dollar fort = or faible).
    Source : corrélation empirique -0.82 DXY/Gold
    """
    return -np.clip(dxy_returns_20d * 500, -100, 100)


def _cot_divergence_signal(commercial_index: float, spec_index: float) -> float:
    """
    Divergence COT Commerciaux vs Large Speculators.
    Grande divergence = explosion imminente de prix.
    Amélioration technique Larry Williams (USA).
    """
    divergence = commercial_index - spec_index   # +100 = Comm très long, Spec très short
    return np.clip(divergence, -100, 100)


def _sgx_premium_signal(sgx_premium_usd: float) -> float:
    """
    Shanghai Gold Exchange premium vs COMEX.
    > $30 = forte demande physique Chine → bullish.
    Source : SGX data + World Gold Council
    """
    if sgx_premium_usd > SGX_PREMIUM_BULLISH_THRESHOLD:
        return min(100, sgx_premium_usd * 2)
    elif sgx_premium_usd > 10:
        return 30.0
    elif sgx_premium_usd < -10:
        return -50.0
    return 0.0


def _petrodollar_signal(oil_returns_5d: float, dxy_5d: float) -> float:
    """
    Pétrodollars en excès cherchent refuge dans l'or.
    Pétrole haut + Dollar faible = pétrodollars → or.
    Source : corrélation pétrole-or via circuit pétrodollar
    """
    if oil_returns_5d > 0.02 and dxy_5d < 0:
        return 60.0
    elif oil_returns_5d > 0.05 and dxy_5d < -0.01:
        return 100.0
    elif oil_returns_5d < -0.03 and dxy_5d > 0.01:
        return -60.0
    return 0.0


def _zar_proxy_signal(zar_returns_5d: float, zar_corr_deviation: float = 0.0) -> float:
    """
    Rand sud-africain (ZAR) comme proxy offre minière.
    ZAR faible = production sous pression → offre basse → or haussier.
    Quand corrélation ZAR/Gold se brise → signal mean-reversion.
    Source : Afrique du Sud = 8e producteur mondial (~90t/an).
    Corrélation normale XAUUSD/USDZAR : -0.83 (CMTrading 2026)
    """
    base = np.clip(-zar_returns_5d * 300, -100, 100)
    # Brisure de corrélation → amplification du signal
    if abs(zar_corr_deviation) > ZAR_CORR_DEVIATION_THRESHOLD:
        base *= 1.4
    return np.clip(base, -100, 100)


def _petrodollar_lag_signal(oil_lag3w_returns: float, gold_momentum_10d: float) -> float:
    """
    Pétrole leads or de ~3 semaines via recyclage pétrodollar.
    Fonds souverains Golfe (ADIA, PIF) recyclent pétrole → or physique.
    Ratio or/pétrole historique : 15.5 (mean-reversion tradable).
    Source : AronGroups + StoneX + analyse pétrodollar 2024-2025
    """
    lag_signal = oil_lag3w_returns - gold_momentum_10d
    return np.clip(lag_signal * 200, -60, 60)


def _krw_stress_signal(krw_zscore: float) -> float:
    """
    Stress KRW → NPS (National Pension Service Corée, 600 Mds$) achète or.
    Quand USD/KRW dépasse +2σ → activation couverture NPS → demande or.
    Source : Bloomberg 2026 — NPS adopte +5% couverture tactique.
    """
    if krw_zscore > KRW_STRESS_ZSCORE:
        return min(60.0 + (krw_zscore - 2.0) * 15, 80.0)
    elif krw_zscore < -KRW_STRESS_ZSCORE:
        return -40.0
    return 0.0


def _yen_divergence_signal(yen_gold_divergence: float) -> float:
    """
    Divergence JPY/Gold : JPY en avance sur or de 2-5 jours.
    Carry Trade Unwind 2024 : USDJPY 161→141 en 3 semaines, or suit après.
    Source : StoneX 2024 USD/JPY Carry Trade analysis.
    Corrélation USDJPY/Gold : -91.8% sur 30 jours (Orbex).
    """
    if yen_gold_divergence > 1.5:
        return 70.0    # JPY déjà monté, gold va suivre → LONG
    elif yen_gold_divergence > 0.8:
        return 35.0
    elif yen_gold_divergence < -1.5:
        return -70.0   # JPY déjà baissé, gold va suivre → SHORT
    elif yen_gold_divergence < -0.8:
        return -35.0
    return 0.0


def session_timing_multiplier(hour_utc: int) -> float:
    """
    Module le poids des signaux selon l'heure asiatique.
    Technique High-Flyer Quant (Chine) — les signaux 02h-05h UTC
    ont le meilleur ratio rendement/risque pendant la session asiatique.
    """
    for (start, end), mult in SESSION_MULTIPLIERS.items():
        if start <= hour_utc < end:
            return mult
    return 0.6


def alpha_decay_halflife(signal_age_hours: float, halflife_h: float = ALPHA_DECAY_HALFLIFE_DEFAULT) -> float:
    """
    Décroissance exponentielle d'un signal avec le temps.
    arXiv 2512.11913 — High-Flyer Quant : chaque signal a une demi-vie.
    Recycler les signaux avant leur mort > attendre qu'ils deviennent faux.
    Demi-vie calibrée pour or 1h : ~12h (signal COT : 48-72h).
    """
    return 0.5 ** (signal_age_hours / halflife_h)


def compute_macro_layer(
    real_rates: float,
    dxy_returns_20d: float,
    commercial_index: float,
    spec_index: float,
    sgx_premium: float = 0.0,
    oil_returns_5d: float = 0.0,
    dxy_returns_5d: float = 0.0,
    zar_returns_5d: float = 0.0,
    zar_corr_deviation: float = 0.0,
) -> NexusLayer:
    """Couche 1 : contexte macro hebdo/journalier."""
    rr   = _real_rates_signal(real_rates)
    dxy  = _dxy_momentum_signal(dxy_returns_20d)
    cot  = _cot_divergence_signal(commercial_index, spec_index)
    sgx  = _sgx_premium_signal(sgx_premium)
    pet  = _petrodollar_signal(oil_returns_5d, dxy_returns_5d)
    zar  = _zar_proxy_signal(zar_returns_5d, zar_corr_deviation)

    score = (
        rr  * W_REAL_RATES   +
        dxy * W_DXY_MOMENTUM +
        cot * W_COT_DIVERGE  +
        sgx * W_SGX_PREMIUM  +
        pet * W_PETRODOLLAR  +
        zar * W_ZAR_PROXY
    )

    return NexusLayer(
        name="MACRO",
        score=np.clip(score, -100, 100),
        components={
            "real_rates": rr, "dxy": dxy, "cot_div": cot,
            "sgx": sgx, "petro": pet, "zar": zar,
        },
    )


# ─────────────────────────────────────────────────────────────
#  COUCHE 2 — FLUX INSTITUTIONNEL
# ─────────────────────────────────────────────────────────────

def _etf_flow_signal(volume_flow_5d: float, volume_flow_20d: float) -> float:
    """
    Flux GLD/IAU — inflows institutionnels vs moyenne.
    volume_flow > 1.5 = inflow exceptionnel = demande forte.
    """
    avg = (volume_flow_5d + volume_flow_20d) / 2
    return np.clip((avg - 1.0) * 200, -100, 100)


def _smart_money_score(
    commercial_index: float,
    etf_flow_signal: float,
    spec_index: float,
) -> float:
    """
    Composite Smart Money = Commerciaux + ETF flows + Large Spec.
    Tous dans le même sens = forte conviction institutionnelle.
    """
    return (commercial_index * 0.4 + etf_flow_signal * 0.35 + spec_index * 0.25)


def _dumb_money_score(retail_long_pct: float) -> float:
    """
    Score retail OANDA inversé → signal contrarian.
    > 75% retail long = probablement faux → bearish pour nous.
    Source : technique SentimenTrader / Ned Davis Research (USA)
    """
    neutral = 0.5
    deviation = retail_long_pct - neutral
    return np.clip(-deviation * 200, -100, 100)


def _yen_safe_haven_signal(usdjpy_corr_20: float) -> float:
    """
    Corrélation roulante 20 barres entre or et USD/JPY (inversée).
    Corrélation fortement négative = les deux safe havens bougent ensemble.
    = risk-off confirmé = signal bullish or.
    Source : Bank of Japan / analyse carry trade yen
    """
    if usdjpy_corr_20 < -0.6:
        return 80.0
    elif usdjpy_corr_20 < -0.3:
        return 40.0
    elif usdjpy_corr_20 > 0.3:
        return -40.0
    return 0.0


def _india_seasonality_signal(month: int, usdinr_returns_5d: float) -> float:
    """
    Saisonnalité demande physique Inde :
    - Oct/Nov = Diwali + Dhanteras = pic achat or
    - Jan/Feb + Mai/Jun = mariages = achat soutenu
    Roupie faible amplifie la demande locale (or en roupies = moins cher rel.)
    Source : World Gold Council India demand reports
    """
    base = 0.0
    if month in INDIA_FESTIVE_MONTHS:
        base = 70.0
    elif month in INDIA_WEDDING_MONTHS:
        base = 40.0

    # Roupie faible → demande locale amplifiée
    if usdinr_returns_5d > 0.01 and base > 0:
        base *= 1.3
    return min(base, 100.0)


def compute_flow_layer(
    volume_flow_5d: float,
    volume_flow_20d: float,
    commercial_index: float,
    spec_index: float,
    retail_long_pct: float = 0.5,
    usdjpy_corr_20: float = 0.0,
    yen_gold_divergence: float = 0.0,
    month: int = 1,
    usdinr_returns_5d: float = 0.0,
    krw_zscore: float = 0.0,
    oil_lag3w_returns: float = 0.0,
    gold_momentum_10d: float = 0.0,
    hour_utc: int = 12,
) -> NexusLayer:
    """
    Couche 2 : qui achète/vend en ce moment ?
    Intègre : ETF flows, Smart/Dumb Money, Yen divergence (JP),
    Inde saisonnalité, NPS Corée, Pétrodollar lag (Moyen-Orient).
    """
    etf    = _etf_flow_signal(volume_flow_5d, volume_flow_20d)
    smart  = _smart_money_score(commercial_index, etf, spec_index)
    dumb   = _dumb_money_score(retail_long_pct)
    yen    = _yen_safe_haven_signal(usdjpy_corr_20)
    yen_div = _yen_divergence_signal(yen_gold_divergence)
    india  = _india_seasonality_signal(month, usdinr_returns_5d)
    krw    = _krw_stress_signal(krw_zscore)
    petro  = _petrodollar_lag_signal(oil_lag3w_returns, gold_momentum_10d)

    # Moduler par la session asiatique (technique High-Flyer Chine)
    asian_mult = session_timing_multiplier(hour_utc)
    yen_div_adjusted = yen_div * asian_mult
    krw_adjusted     = krw    * asian_mult

    score = (
        etf             * 0.20 +
        smart           * 0.25 +
        dumb            * 0.15 +
        yen             * 0.10 +
        yen_div_adjusted * 0.10 +
        india           * 0.08 +
        krw_adjusted    * 0.07 +
        petro           * 0.05
    )

    return NexusLayer(
        name="FLOW",
        score=np.clip(score, -100, 100),
        components={
            "etf": etf, "smart": smart, "dumb": dumb,
            "yen_corr": yen, "yen_div": yen_div_adjusted,
            "india": india, "krw": krw_adjusted, "petro": petro,
        },
    )


# ─────────────────────────────────────────────────────────────
#  COUCHE 3 — RÉGIME & TIMING
# ─────────────────────────────────────────────────────────────

def _regime_momentum_signal(
    prev_regime: int,
    curr_regime: int,
    price_direction: int,   # +1 montée, -1 baisse
) -> float:
    """
    MA CONTRIBUTION ORIGINALE : la transition de régime HMM précède le prix.
    RANGE→TREND = fort signal directionnel.
    TREND→CHAOS = sortie immédiate.
    """
    RANGE, TREND, CHAOS = 0, 1, 2

    if prev_regime == RANGE and curr_regime == TREND:
        return 90.0 * price_direction   # Signal fort : début de tendance
    elif prev_regime == TREND and curr_regime == TREND:
        return 40.0 * price_direction   # Tendance en cours
    elif prev_regime == RANGE and curr_regime == RANGE:
        return 0.0                      # Pas de signal en range
    elif curr_regime == CHAOS:
        return -999.0                   # Signal d'urgence : ne pas trader
    elif prev_regime == TREND and curr_regime == RANGE:
        return -30.0 * price_direction  # Fin de tendance
    return 0.0


def _alpha_decay(raw_score: float, signal_age_hours: float) -> float:
    """
    Technique High-Flyer Quant (Chine, 幻方量化).
    Les signaux se dégradent exponentiellement avec le temps.
    Demi-vie = 24h → signal de 48h vaut 25% de sa valeur initiale.
    """
    decay = 0.5 ** (signal_age_hours / ALPHA_DECAY_HALFLIFE_H)
    return raw_score * decay


def _turtle_signal(
    price: float,
    high_20d: float,
    low_20d: float,
    regime: int,
) -> float:
    """
    Turtle Trading modernisé : cassage 20j high/low.
    SEULEMENT en régime TREND (filtre HMM) pour éviter faux cassages.
    Source : Richard Dennis & William Eckhardt (USA, 1983)
    """
    if regime != 1:   # seulement TREND
        return 0.0
    if price > high_20d:
        return 70.0   # Cassage haussier → LONG
    elif price < low_20d:
        return -70.0  # Cassage baissier → SHORT
    return 0.0


def _lunar_signal(timestamp: datetime) -> float:
    """
    Cycle lunaire — corrélation or avec nouvelle lune.
    Achat traditionnel Inde/Chine autour nouvelle lune.
    Source : Batchelor & Ramyar (City University London)
    """
    # J2000 epoch : nouvelle lune le 6 Jan 2000
    j2000_new_moon = datetime(2000, 1, 6, tzinfo=timezone.utc)
    delta_days = (timestamp.replace(tzinfo=timezone.utc) - j2000_new_moon).total_seconds() / 86400
    phase_days = delta_days % LUNAR_PERIOD_DAYS
    days_from_new_moon = min(phase_days, LUNAR_PERIOD_DAYS - phase_days)

    if days_from_new_moon <= 2:
        return 30.0   # ±2 jours nouvelle lune = achats traditionnels
    elif days_from_new_moon >= 12 and days_from_new_moon <= 16:
        return -15.0  # Pleine lune = légère pression vente
    return 0.0


def compute_regime_layer(
    prev_regime: int,
    curr_regime: int,
    price_direction: int,
    price: float,
    high_20d: float,
    low_20d: float,
    multitf_bullish: float,
    multitf_bearish: float,
    cot_signal_age_hours: float = 0.0,
    cot_raw_score: float = 0.0,
    timestamp: Optional[datetime] = None,
) -> NexusLayer:
    """Couche 3 : est-ce le bon moment pour entrer ?"""
    regime_mom  = _regime_momentum_signal(prev_regime, curr_regime, price_direction)

    # Court-circuit : régime CHAOS = aucun trade possible
    if regime_mom <= -999:
        return NexusLayer(
            name="REGIME",
            score=-100.0,
            components={"chaos": True},
        )

    multitf = (multitf_bullish - multitf_bearish) * 100
    turtle  = _turtle_signal(price, high_20d, low_20d, curr_regime)
    cot_t   = _alpha_decay(cot_raw_score, cot_signal_age_hours)
    lunar   = _lunar_signal(timestamp) if timestamp else 0.0

    score = (
        regime_mom * W_REGIME_MOMENTUM +
        multitf    * W_MULTITF         +
        turtle     * W_TURTLE          +
        cot_t      * W_COT_TIMING      +
        lunar      * 0.05
    )

    return NexusLayer(
        name="REGIME",
        score=np.clip(score, -100, 100),
        components={
            "regime_momentum": regime_mom,
            "multitf": multitf,
            "turtle": turtle,
            "cot_decayed": cot_t,
            "lunar": lunar,
        },
    )


# ─────────────────────────────────────────────────────────────
#  COUCHE 4 — PRÉCISION D'ENTRÉE
# ─────────────────────────────────────────────────────────────

def detect_judas_swing(
    hour: int,
    minute: int,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    asian_high: float,
    asian_low: float,
) -> Optional[str]:
    """
    Judas Swing — faux breakout en début London pour collecter les stops.
    Source : Michael Huddleston / ICT, vidéos publiques 2022-2023
    """
    is_london_open = (hour == 8 and minute < 30)
    if not is_london_open:
        return None

    # Spike au-dessus de l'Asian High → retomb en dessous = Judas SHORT
    if bar_high > asian_high * 1.0005 and bar_close < asian_high:
        return "JUDAS_SHORT"

    # Spike en dessous de l'Asian Low → remonte au-dessus = Judas LONG
    if bar_low < asian_low * 0.9995 and bar_close > asian_low:
        return "JUDAS_LONG"

    return None


def detect_liquidity_hunt(
    price: float,
    recent_highs: list[float],
    recent_lows: list[float],
    bar_direction: int,     # +1 montée, -1 baisse
    tolerance: float = 0.002,
) -> Optional[str]:
    """
    Détection de chasse de liquidité (Liquidity Sweep).
    Les institutionnels chassent les stops accumulés aux equal highs/lows.
    Après la chasse → retournement = notre signal d'entrée.
    """
    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return None

    equal_high_zone = max(recent_highs[-5:])
    equal_low_zone  = min(recent_lows[-5:])

    price_above_highs = price > equal_high_zone * (1 + tolerance)
    price_below_lows  = price < equal_low_zone  * (1 - tolerance)

    if price_above_highs and bar_direction == -1:
        return "HUNT_SHORT"
    elif price_below_lows and bar_direction == +1:
        return "HUNT_LONG"

    return None


def compute_entry_trigger(
    hour: int,
    minute: int,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    asian_high: float,
    asian_low: float,
    recent_highs: list[float],
    recent_lows: list[float],
    bar_direction: int,
    is_london: bool,
    is_ny: bool,
    fvg_bull: float,
    fvg_bear: float,
) -> Optional[str]:
    """
    Détermine le trigger d'entrée précis.
    Priorité : Judas Swing > Liquidity Hunt > FVG ICT
    """
    if not (is_london or is_ny):
        return None

    # Filtre : pas les 30 premières minutes de London (faux mouvements)
    if hour == 8 and minute < 30:
        judas = detect_judas_swing(
            hour, minute, bar_high, bar_low, bar_close, asian_high, asian_low
        )
        if judas:
            return "LONG" if judas == "JUDAS_LONG" else "SHORT"
        return None

    # Liquidity Hunt
    hunt = detect_liquidity_hunt(
        bar_close, recent_highs, recent_lows, bar_direction
    )
    if hunt:
        return "LONG" if hunt == "HUNT_LONG" else "SHORT"

    # ICT Fair Value Gap comme dernier filtre
    if fvg_bull > 0:
        return "LONG"
    elif fvg_bear > 0:
        return "SHORT"

    return None


# ─────────────────────────────────────────────────────────────
#  SIGNAL FINAL NEXUS
# ─────────────────────────────────────────────────────────────

def _dynamic_kelly(composite: float, base_risk: float = 0.005) -> float:
    """
    Kelly dynamique — inspiré de High-Flyer Quant (Chine).
    Plus le signal est fort, plus on risque (dans une limite stricte).
    """
    confidence = abs(composite) / 100.0
    dynamic = base_risk * (1 + confidence * 2)
    return min(dynamic, 0.015)   # Cap absolu à 1.5%


def generate_nexus_signal(
    macro: NexusLayer,
    flow: NexusLayer,
    regime: NexusLayer,
    entry_trigger: Optional[str],
) -> NexusSignal:
    """
    Agrège les 4 couches en signal final avec confiance et sizing dynamique.
    """
    composite = (
        regime.score * W_REGIME_FINAL +
        flow.score   * W_FLOW_FINAL   +
        macro.score  * W_MACRO_FINAL
    )

    direction = "FLAT"
    if entry_trigger is not None:
        if composite >= NEXUS_LONG_THRESHOLD and entry_trigger == "LONG":
            direction = "LONG"
        elif composite <= NEXUS_SHORT_THRESHOLD and entry_trigger == "SHORT":
            direction = "SHORT"

    confidence   = abs(composite) / 100.0
    dynamic_risk = _dynamic_kelly(composite) if direction != "FLAT" else 0.0

    sig = NexusSignal(
        direction=direction,
        composite=composite,
        confidence=confidence,
        macro=macro,
        flow=flow,
        regime=regime,
        entry_trigger=entry_trigger,
        dynamic_risk_pct=dynamic_risk,
    )

    if sig.is_valid():
        logger.success(sig.summary())
    return sig


# ─────────────────────────────────────────────────────────────
#  FONCTION PRINCIPALE : à appeler depuis build_features()
# ─────────────────────────────────────────────────────────────

def nexus_composite_score(row: pd.Series, prev_regime: int) -> dict:
    """
    Prend une ligne de features déjà construites et retourne les scores NEXUS.
    Appeler après build_features() pour chaque barre.
    """
    ts = row.name if isinstance(row.name, datetime) else datetime.now(timezone.utc)

    macro = compute_macro_layer(
        real_rates        = row.get("real_rates", 0.0),
        dxy_returns_20d   = row.get("dxy_returns", 0.0),
        commercial_index  = row.get("cot_commercial_index", 50.0),
        spec_index        = row.get("cot_spec_index", 50.0),
        sgx_premium       = row.get("sgx_premium", 0.0),
        oil_returns_5d    = row.get("oil_returns_5d", 0.0),
        dxy_returns_5d    = row.get("dxy_returns_5d", 0.0),
        zar_returns_5d    = row.get("zar_returns_5d", 0.0),
    )

    flow = compute_flow_layer(
        volume_flow_5d    = row.get("volume_flow_5d", 1.0),
        volume_flow_20d   = row.get("volume_flow_20d", 1.0),
        commercial_index  = row.get("cot_commercial_index", 50.0),
        spec_index        = row.get("cot_spec_index", 50.0),
        retail_long_pct   = row.get("retail_long_pct", 0.5),
        usdjpy_corr_20    = row.get("usdjpy_corr_20", 0.0),
        month             = ts.month,
        usdinr_returns_5d = row.get("usdinr_returns_5d", 0.0),
    )

    curr_regime   = int(row.get("regime_id", 0))
    price_dir     = int(np.sign(row.get("returns_1h", 0.0)))

    regime = compute_regime_layer(
        prev_regime        = prev_regime,
        curr_regime        = curr_regime,
        price_direction    = price_dir,
        price              = row.get("close", 1900.0),
        high_20d           = row.get("high_20d", row.get("close", 1900.0)),
        low_20d            = row.get("low_20d",  row.get("close", 1900.0)),
        multitf_bullish    = row.get("multitf_bullish_confluence", 0.0),
        multitf_bearish    = row.get("multitf_bearish_confluence", 0.0),
        cot_signal_age_hours = row.get("cot_signal_age_hours", 0.0),
        cot_raw_score      = row.get("cot_commercial_index", 50.0) - 50,
        timestamp          = ts,
    )

    return {
        "nexus_macro":   macro.score,
        "nexus_flow":    flow.score,
        "nexus_regime":  regime.score,
        "nexus_composite": float(
            regime.score * W_REGIME_FINAL +
            flow.score   * W_FLOW_FINAL   +
            macro.score  * W_MACRO_FINAL
        ),
        "nexus_chaos_flag": 1 if regime.components.get("chaos") else 0,
    }
