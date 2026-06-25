"""
Collecte des données gold depuis plusieurs sources.
Source principale : yfinance (gratuit, pas de clé API)
Source future : MetaAPI (connexion MT5 Exness/Pepperstone)
"""
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from loguru import logger
from src.config import SYMBOL, FRED_SERIES, FRED_API_KEY, ROOT_DIR
from src.database import save_gold_prices

yf.set_tz_cache_location(str(ROOT_DIR / ".cache" / "yfinance"))


def fetch_gold_yfinance(timeframe: str = "1h", days: int = 730) -> pd.DataFrame:
    """
    Télécharge les prix gold depuis Yahoo Finance.
    Gratuit, aucune clé API requise.

    timeframe : "1m" | "5m" | "15m" | "1h" | "4h" | "1d"
    days      : nombre de jours d'historique
    """
    # Mapping vers les intervalles yfinance
    interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m",
        "1h": "1h", "4h": "4h", "1d": "1d"
    }
    yf_interval = interval_map.get(timeframe, "1h")

    # yfinance limite l'historique selon l'intervalle
    period_map = {
        "1m": "7d", "5m": "60d", "15m": "60d",
        "1h": "730d", "4h": "730d", "1d": "10y"
    }
    period = period_map.get(timeframe, "730d")

    logger.info(f"📥 Téléchargement gold {timeframe} ({period}) depuis yfinance...")

    ticker = yf.Ticker(SYMBOL)
    df = ticker.history(period=period, interval=yf_interval, auto_adjust=True)

    if df.empty:
        logger.error(f"❌ Aucune donnée reçue pour {SYMBOL}")
        return pd.DataFrame()

    # Nettoyer et renommer
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.dropna(inplace=True)

    logger.success(f"✅ {len(df)} bougies {timeframe} téléchargées — Gold {df['Close'].iloc[-1]:.2f}$")
    return df


def collect_and_store_gold(timeframe: str = "1h") -> int:
    """Télécharge et stocke les prix gold en base."""
    df = fetch_gold_yfinance(timeframe=timeframe)
    if df.empty:
        return 0
    return save_gold_prices(df, timeframe=timeframe, source="yfinance")


def fetch_fred_macro() -> dict:
    """
    Télécharge les données macro depuis FRED.
    Requiert FRED_API_KEY dans .env
    Gratuit sur https://fred.stlouisfed.org/docs/api/api_key.html
    """
    if not FRED_API_KEY:
        logger.warning("⚠️ FRED_API_KEY non configuré — données macro ignorées")
        return {}

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        results = {}

        for series_id, name in FRED_SERIES.items():
            try:
                series = fred.get_series(series_id, observation_start="2020-01-01")
                results[series_id] = series
                logger.debug(f"  📈 {series_id} ({name}) : dernière valeur {series.iloc[-1]:.4f}")
            except Exception as e:
                logger.warning(f"  ⚠️ {series_id} non disponible : {e}")

        logger.success(f"✅ {len(results)} séries macro FRED téléchargées")
        return results

    except ImportError:
        logger.error("❌ fredapi non installé : pip install fredapi")
        return {}


def fetch_cot_gold() -> pd.DataFrame:
    """
    Télécharge le rapport COT (Commitment of Traders) pour le gold.
    Publié chaque vendredi par la CFTC.
    Source : CFTC.gov — gratuit, public.

    Retourne un DataFrame avec les positions commerciales/spéculatives.
    """
    import requests
    import io
    import zipfile

    logger.info("📥 Téléchargement rapport COT CFTC (gold futures)...")

    # URL du rapport COT Legacy Commodities (contient l'or/gold futures)
    # La CFTC ne publie les fichiers annuels qu'en fin d'année → fallback sur l'année précédente
    year = datetime.now().year
    # URL officielle CFTC — Futures Only Report (Legacy, contient l'or)
    # Source : https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm
    response = None
    for y in [year, year - 1, year - 2]:
        url = f"https://www.cftc.gov/files/dea/history/deacot{y}.zip"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                response = r
                logger.info(f"  COT : fichier deacot{y}.zip trouvé")
                break
        except Exception:
            continue
    if response is None:
        logger.error(f"❌ Rapport COT CFTC introuvable pour les années {year}, {year-1}, {year-2}")
        return pd.DataFrame()

    try:
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            filename = z.namelist()[0]
            with z.open(filename) as f:
                df_full = pd.read_csv(f, low_memory=False)

        # Normaliser les noms de colonnes (espaces → underscores, trim)
        df_full.columns = df_full.columns.str.strip().str.replace(r'[\s\-/()]+', '_', regex=True).str.rstrip('_')

        # Trouver la colonne marché (nom variable selon les versions CFTC)
        market_col = next((c for c in df_full.columns if 'Market' in c and 'Exchange' in c), None)
        if market_col is None:
            logger.error(f"❌ Colonne marché introuvable. Colonnes disponibles : {list(df_full.columns[:10])}")
            return pd.DataFrame()

        # Filtrer uniquement le contrat principal Gold COMEX.
        gold_mask = df_full[market_col].str.strip().eq("GOLD - COMMODITY EXCHANGE INC.")
        df_gold = df_full[gold_mask].copy()

        if df_gold.empty:
            logger.warning("⚠️ Pas de données gold dans le rapport COT")
            return pd.DataFrame()

        # Trouver les colonnes clés de façon flexible (exclude évite les faux positifs)
        def find_col(df, *keywords, exclude=None):
            for col in df.columns:
                col_l = col.lower()
                if all(k.lower() in col_l for k in keywords):
                    if exclude is None or not any(e.lower() in col_l for e in exclude):
                        return col
            return None

        date_col    = find_col(df_gold, 'Date') or find_col(df_gold, 'date')
        comm_long   = find_col(df_gold, 'Commercial', 'Long', exclude=['Non'])
        comm_short  = find_col(df_gold, 'Commercial', 'Short', exclude=['Non'])
        ncomm_long  = find_col(df_gold, 'Noncommercial', 'Long')
        ncomm_short = find_col(df_gold, 'Noncommercial', 'Short')
        nrept_long  = find_col(df_gold, 'Nonreportable', 'Long') or find_col(df_gold, 'NonRept', 'Long')
        nrept_short = find_col(df_gold, 'Nonreportable', 'Short') or find_col(df_gold, 'NonRept', 'Short')

        logger.debug(f"  Colonnes COT : date={date_col}, comm_long={comm_long}, ncomm_long={ncomm_long}")

        # Parser la date : format YYMMDD (ex: 260609 = 2026-06-09) ou texte standard
        raw_date = df_gold[date_col]
        if pd.api.types.is_numeric_dtype(raw_date) or str(raw_date.iloc[0]).isdigit():
            report_date = pd.to_datetime(raw_date.astype(str).str.zfill(6), format='%y%m%d')
        else:
            report_date = pd.to_datetime(raw_date)

        df_cot = pd.DataFrame({
            "report_date":        report_date,
            "commercial_long":    pd.to_numeric(df_gold[comm_long], errors="coerce"),
            "commercial_short":   pd.to_numeric(df_gold[comm_short], errors="coerce"),
            "large_spec_long":    pd.to_numeric(df_gold[ncomm_long], errors="coerce"),
            "large_spec_short":   pd.to_numeric(df_gold[ncomm_short], errors="coerce"),
            "small_trader_long":  pd.to_numeric(df_gold[nrept_long] if nrept_long else 0, errors="coerce"),
            "small_trader_short": pd.to_numeric(df_gold[nrept_short] if nrept_short else 0, errors="coerce"),
        })

        df_cot["commercial_net"] = df_cot["commercial_long"] - df_cot["commercial_short"]
        df_cot["large_spec_net"] = df_cot["large_spec_long"] - df_cot["large_spec_short"]

        # COT Index : normalisation sur 52 semaines (méthode Larry Williams)
        df_cot = df_cot.sort_values("report_date").reset_index(drop=True)
        rolling = df_cot["commercial_net"].rolling(52, min_periods=10)
        df_cot["commercial_index"] = (
            (df_cot["commercial_net"] - rolling.min()) /
            (rolling.max() - rolling.min()) * 100
        ).round(1)

        # Signal COT
        df_cot["cot_signal"] = "NEUTRAL"
        df_cot.loc[df_cot["commercial_index"] > 75, "cot_signal"] = "BULLISH"
        df_cot.loc[df_cot["commercial_index"] < 25, "cot_signal"] = "BEARISH"

        logger.success(f"✅ COT gold : {len(df_cot)} semaines — dernier signal : {df_cot['cot_signal'].iloc[-1]} (index={df_cot['commercial_index'].iloc[-1]:.1f})")
        return df_cot

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Erreur téléchargement COT : {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ Erreur parsing COT : {e}")
        return pd.DataFrame()


def fetch_etf_flows(days: int = 730) -> pd.DataFrame:
    """
    Flux des ETFs or GLD et IAU — signal institutionnel avancé.

    Sources :
      - World Gold Council (UK) : ETF flows comme indicateur de demande institutionnelle
      - stefan-jansen/machine-learning-for-trading (GitHub, 9k étoiles) :
        GLD/IAU comme features prédictifs sur gold futures
      - Technique : inflows forts sur 5 jours → demande institutionnelle → prix suit dans 24-48h

    La Chine est le 1er acheteur mondial d'or mais les ETFs reflètent la demande
    occidentale (US, Europe). Ensemble ils couvrent 80%+ de la demande mondiale.
    """
    tickers = {
        "GLD": "SPDR Gold Shares (le plus grand ETF or au monde)",
        "IAU": "iShares Gold Trust (2e plus grand ETF or)",
        "GDX": "VanEck Gold Miners ETF (proxy demande minières = leading indicator)",
    }

    result = {}
    for symbol, desc in tickers.items():
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d", interval="1d", auto_adjust=True)
            if not df.empty:
                df.index = pd.to_datetime(df.index, utc=True)
                result[symbol] = df[["Close", "Volume"]].rename(columns={
                    "Close":  f"{symbol}_close",
                    "Volume": f"{symbol}_volume",
                })
                logger.debug(f"  📊 {symbol} ({desc}) : {len(df)} jours")
        except Exception as e:
            logger.warning(f"  ⚠️ ETF {symbol} non disponible : {e}")

    if not result:
        return pd.DataFrame()

    # Fusionner tous les ETFs sur le même index daily
    df_etf = pd.concat(result.values(), axis=1)
    df_etf.sort_index(inplace=True)

    # Calculer les signaux de flux
    for sym in ["GLD", "IAU"]:
        vol_col = f"{sym}_volume"
        if vol_col in df_etf.columns:
            # Volume relatif vs moyenne 20 jours = proxy de flux entrants/sortants
            df_etf[f"{sym}_flow_signal"] = (
                df_etf[vol_col] / df_etf[vol_col].rolling(20).mean()
            ).fillna(1.0)

    # Signal combiné : moyenne des deux ETFs
    if "GLD_flow_signal" in df_etf.columns and "IAU_flow_signal" in df_etf.columns:
        df_etf["etf_combined_flow"] = (
            df_etf["GLD_flow_signal"] + df_etf["IAU_flow_signal"]
        ) / 2

    logger.success(f"✅ ETF flows : {len(df_etf)} jours (GLD + IAU + GDX)")
    return df_etf


def fetch_oanda_sentiment() -> dict:
    """
    Positionnement retail OANDA sur XAU/USD — signal contrarian.

    Source : OANDA Open Positions API (gratuit, public)
    Technique : Quand 75%+ des retail sont LONG → signal SHORT (fade the crowd)
    Logique : Les retail traders ont systématiquement tort aux extrêmes de marché
              (confirmé par CFTC, études Barclays 2015, Man AHL UK)

    Retourne : dict avec 'long_pct', 'short_pct', 'contrarian_signal'
    """
    try:
        import requests
        # OANDA public sentiment (pas d'auth requise pour les données agrégées)
        url = "https://www.oanda.com/bvi-en/lab-education/tools/sentiment/"
        # Note : endpoint REST direct via oandapyV20 nécessite un compte demo gratuit
        # Pour l'instant on retourne une valeur neutre par défaut
        logger.debug("  OANDA sentiment : utilisation valeur neutre (configurer oandapyV20)")
        return {
            "long_pct": 0.5,
            "short_pct": 0.5,
            "contrarian_signal": 0.0,  # -1=short, 0=neutre, +1=long
        }
    except Exception as e:
        logger.warning(f"  ⚠️ OANDA sentiment non disponible : {e}")
        return {"long_pct": 0.5, "short_pct": 0.5, "contrarian_signal": 0.0}


def run_initial_collection():
    """
    Collecte initiale complète au premier lancement.
    Télécharge 2 ans de données 1H et Daily.
    """
    logger.info("🚀 Collecte initiale des données gold...")

    # Données de prix (plusieurs timeframes)
    for tf in ["1d", "4h", "1h", "15m"]:
        n = collect_and_store_gold(timeframe=tf)
        logger.info(f"  {tf}: {n} bougies stockées")
        time.sleep(2)  # Pause pour ne pas spammer Yahoo Finance

    # COT
    df_cot = fetch_cot_gold()
    if not df_cot.empty:
        _save_cot(df_cot)

    # Macro FRED (si clé disponible)
    macro = fetch_fred_macro()
    if macro:
        _save_macro(macro)

    logger.success("✅ Collecte initiale terminée")


def _save_cot(df_cot: pd.DataFrame):
    """Sauvegarde les données COT en base."""
    from sqlalchemy import text
    from src.database import get_engine

    sql = text("""
        INSERT INTO cot_data
            (report_date, commercial_long, commercial_short, commercial_net,
             commercial_index, large_spec_long, large_spec_short, large_spec_net,
             small_trader_long, small_trader_short, cot_signal)
        VALUES
            (:report_date, :commercial_long, :commercial_short, :commercial_net,
             :commercial_index, :large_spec_long, :large_spec_short, :large_spec_net,
             :small_trader_long, :small_trader_short, :cot_signal)
        ON CONFLICT DO NOTHING
    """)
    records = df_cot.to_dict(orient="records")
    with get_engine().begin() as conn:
        conn.execute(sql, records)
    logger.info(f"💾 {len(records)} semaines COT sauvegardées")


def _save_macro(macro: dict):
    """Sauvegarde les données macro FRED en base."""
    from sqlalchemy import text
    from src.database import get_engine

    sql = text("""
        INSERT INTO macro_data (time, series_id, value, series_name)
        VALUES (:time, :series_id, :value, :series_name)
        ON CONFLICT DO NOTHING
    """)
    records = []
    for series_id, series in macro.items():
        for date, value in series.items():
            if pd.notna(value):
                records.append({
                    "time": date, "series_id": series_id,
                    "value": float(value),
                    "series_name": FRED_SERIES.get(series_id, series_id)
                })

    with get_engine().begin() as conn:
        conn.execute(sql, records)
    logger.info(f"💾 {len(records)} points macro FRED sauvegardés")
