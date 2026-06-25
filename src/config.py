"""
Configuration centrale du GoldBot.
Toutes les constantes et paramètres passent par ici.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Base de données ───────────────────────────────────────
DB_HOST     = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))
DB_USER     = os.getenv("POSTGRES_USER", "trader")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "goldbot_secure_2026")
DB_NAME     = os.getenv("POSTGRES_DB", "goldbot")
DB_URL      = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── APIs externes ─────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
FRED_API_KEY     = os.getenv("FRED_API_KEY", "")
METAAPI_TOKEN    = os.getenv("METAAPI_TOKEN", "")
METAAPI_ACCOUNT  = os.getenv("METAAPI_ACCOUNT_ID", "")

# ── NVIDIA NIM ────────────────────────────────────────────
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "http://localhost:8000/v1")
NIM_API_KEY  = os.getenv("NIM_API_KEY", "nim-local-key")
NIM_MODEL_SENTIMENT = "deepseek-r1"
NIM_MODEL_CODER     = "deepseek-coder"

# ── Marché ────────────────────────────────────────────────
SYMBOL          = "GC=F"        # Ticker yfinance pour Gold Futures
SYMBOL_MT5      = "XAUUSDm"     # Ticker MT5 Exness Demo (m = micro)
SYMBOL_DISPLAY  = "XAU/USD"

# ── Paramètres de trading ─────────────────────────────────
TRADING_MODE        = os.getenv("TRADING_MODE", "paper")     # paper | live
MAX_RISK_PER_TRADE  = float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))   # 2%
MAX_DAILY_DRAWDOWN  = float(os.getenv("MAX_DAILY_DRAWDOWN", "0.05"))   # 5%
MIN_SIGNAL_SCORE    = float(os.getenv("MIN_SIGNAL_SCORE", "52"))       # aligné sur LGBM_THRESHOLD_LONG × 100
ATR_SL_MULTIPLIER   = 2.0       # Stop Loss = ATR × 2
ATR_TP_MULTIPLIER   = 3.0       # Take Profit par défaut = ATR × 3
ATR_TP_TREND        = 5.0       # TP en régime TREND → laisser courir la tendance
ATR_TP_RANGE        = 2.0       # TP en régime RANGE → sortir vite avant retournement
KELLY_FRACTION      = 0.25      # Fraction Kelly conservatrice

# ── Coûts réels du marché (pour backtest réaliste) ────────
SPREAD_USD          = 1.5       # Spread XAU/USD Exness pendant sessions actives (London/NY)
SLIPPAGE_USD        = 1.0       # Slippage moyen sur SL en dollars
SWAP_PER_NIGHT_USD  = 2.5       # Frais de swap par nuit (or = swap négatif)

# ── Filtre sessions actives uniquement ────────────────────
TRADE_SESSIONS_ONLY = True      # True = seulement London (8h-17h GMT) et NY (13h-22h GMT)

# ── Régimes HMM ──────────────────────────────────────────
REGIME_NAMES = {0: "RANGE", 1: "TREND", 2: "CHAOS"}
REGIME_RANGE = 0
REGIME_TREND = 1
REGIME_CHAOS = 2
HMM_N_STATES = 3
HMM_LOOKBACK  = 500             # Barres pour entraîner le HMM

# ── Features LightGBM ─────────────────────────────────────
LGBM_FEATURES = [
    # Prix et momentum
    "returns_1h", "returns_4h", "returns_1d",
    "volatility_20", "atr_14", "atr_ratio",
    # Indicateurs techniques
    "williams_r_14", "williams_r_28",
    "ema_cross_9_21", "ema_cross_21_50", "price_vs_ema200",
    # Volume et flux institutionnels (Wyckoff + ETF flows)
    "volume_ratio", "volume_trend",
    "volume_flow_5d", "volume_flow_20d",
    "ad_flow_signal", "volume_momentum_3d",
    # COT CFTC
    "cot_commercial_net", "cot_commercial_index",
    # ICT Smart Money
    "fvg_bull", "fvg_bear",
    # Sessions
    "is_london_session", "is_ny_session", "is_overlap",
    # Macro
    "dxy_returns", "real_rates", "vix_level",
    # Multi-timeframe (nouveau — stefan-jansen + Man AHL)
    "trend_4h_bullish", "price_vs_ema50_4h",
    "trend_daily_bullish", "golden_cross_1d", "death_cross_1d",
    "multitf_bullish_confluence", "multitf_bearish_confluence",
    # Régime HMM
    "regime_id", "regime_confidence",
]
LGBM_TARGET_HORIZON = 4         # Prédire la direction dans 4 heures
LGBM_THRESHOLD_LONG  = 0.52    # Réduit : AUC~0.55 → proba dépasse rarement 0.60
LGBM_THRESHOLD_SHORT = 0.48    # Symétrique

# ── Sessions de marché (GMT) ──────────────────────────────
SESSION_ASIAN_START  = 0        # 00h GMT
SESSION_ASIAN_END    = 8        # 08h GMT
SESSION_LONDON_START = 8        # 08h GMT
SESSION_LONDON_END   = 17       # 17h GMT
SESSION_NY_START     = 13       # 13h GMT
SESSION_NY_END       = 22       # 22h GMT

# ── Sources de données FRED ───────────────────────────────
FRED_SERIES = {
    "DFF":      "Fed Funds Rate",
    "T10Y2Y":   "Courbe des taux (10Y-2Y)",
    "CPIAUCSL": "Inflation CPI",
    "DTWEXBGS": "Dollar Index",
    "M2SL":     "Masse monétaire M2",
}

# ── Entraînement du modèle ────────────────────────────────
RETRAIN_DAY         = 6         # Dimanche (0=lundi, 6=dimanche)
RETRAIN_HOUR        = 23        # 23h
DRIFT_SHARPE_DROP   = 0.30      # Si Sharpe baisse de 30% → retrain d'urgence
BACKTEST_MONTHS     = 6         # Mois de données pour backtester avant déploiement
MIN_SHARPE_DEPLOY   = 0.8       # Sharpe minimum pour déployer le nouveau modèle

# ── Chemins ───────────────────────────────────────────────
import pathlib
ROOT_DIR    = pathlib.Path(__file__).parent.parent
SRC_DIR     = ROOT_DIR / "src"
MODELS_DIR  = ROOT_DIR / "src" / "models"
LOGS_DIR    = ROOT_DIR / "logs"
