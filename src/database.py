"""
Connexion et opérations TimescaleDB.
Point d'entrée unique pour toutes les lectures/écritures en base.
"""
import pandas as pd
from sqlalchemy import create_engine, text
from loguru import logger
from src.config import DB_URL


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, pool_pre_ping=True, pool_size=5)
    return _engine


def test_connection() -> bool:
    try:
        with get_engine().connect() as conn:
            result = conn.execute(text("SELECT version()")).fetchone()
            logger.success(f"✅ TimescaleDB connecté : {result[0][:50]}...")
            return True
    except Exception as e:
        logger.error(f"❌ Connexion base de données échouée : {e}")
        return False


def save_gold_prices(df: pd.DataFrame, timeframe: str = "1h", source: str = "yfinance"):
    """
    Sauvegarde un DataFrame de prix OHLCV dans gold_prices.
    df doit avoir les colonnes : Open, High, Low, Close, Volume (index = datetime)
    """
    if df.empty:
        logger.warning("DataFrame vide — rien à sauvegarder")
        return 0

    records = []
    for ts, row in df.iterrows():
        records.append({
            "time":      ts,
            "open":      float(row["Open"]),
            "high":      float(row["High"]),
            "low":       float(row["Low"]),
            "close":     float(row["Close"]),
            "volume":    float(row.get("Volume", 0) or 0),
            "source":    source,
            "timeframe": timeframe,
        })

    insert_sql = text("""
        INSERT INTO gold_prices (time, open, high, low, close, volume, source, timeframe)
        VALUES (:time, :open, :high, :low, :close, :volume, :source, :timeframe)
        ON CONFLICT DO NOTHING
    """)

    with get_engine().begin() as conn:
        conn.execute(insert_sql, records)

    logger.info(f"💾 {len(records)} bougies {timeframe} sauvegardées ({source})")
    return len(records)


def load_gold_prices(timeframe: str = "1h", days: int = 365) -> pd.DataFrame:
    """
    Charge les prix gold depuis la base pour les N derniers jours.
    Retourne un DataFrame avec index datetime et colonnes OHLCV.
    """
    query = text("""
        SELECT time, open, high, low, close, volume
        FROM gold_prices
        WHERE timeframe = :tf
          AND time > NOW() - INTERVAL ':days days'
        ORDER BY time ASC
    """)
    # sqlalchemy ne supporte pas :days dans INTERVAL, on formate manuellement
    query = text(f"""
        SELECT time, open, high, low, close, volume
        FROM gold_prices
        WHERE timeframe = :tf
          AND time > NOW() - INTERVAL '{days} days'
        ORDER BY time ASC
    """)

    with get_engine().connect() as conn:
        df = pd.read_sql(query, conn, params={"tf": timeframe}, index_col="time", parse_dates=["time"])

    logger.info(f"📊 {len(df)} bougies {timeframe} chargées ({days} jours)")
    return df


def save_signal(signal: dict):
    """Sauvegarde un signal LightGBM en base."""
    import json
    sql = text("""
        INSERT INTO signals
            (time, score, direction, regime, williams_r, cot_index,
             fvg_detected, london_session, pattern_found, sentiment_score, raw_features)
        VALUES
            (:time, :score, :direction, :regime, :williams_r, :cot_index,
             :fvg_detected, :london_session, :pattern_found, :sentiment_score, :raw_features)
    """)
    signal["raw_features"] = json.dumps(signal.get("raw_features", {}))
    with get_engine().begin() as conn:
        conn.execute(sql, signal)
    logger.debug(f"📡 Signal sauvegardé : {signal['direction']} score={signal['score']:.1f}")


def save_regime(time, regime_name: str, regime_id: int, confidence: float, timeframe: str = "1h"):
    """Sauvegarde un régime HMM détecté."""
    sql = text("""
        INSERT INTO market_regimes (time, regime, regime_id, confidence, timeframe)
        VALUES (:time, :regime, :regime_id, :confidence, :timeframe)
        ON CONFLICT DO NOTHING
    """)
    with get_engine().begin() as conn:
        conn.execute(sql, {
            "time": time, "regime": regime_name,
            "regime_id": regime_id, "confidence": confidence,
            "timeframe": timeframe
        })


def get_last_regime(timeframe: str = "1h") -> dict:
    """Retourne le dernier régime détecté."""
    sql = text("""
        SELECT regime, regime_id, confidence, time
        FROM market_regimes
        WHERE timeframe = :tf
        ORDER BY time DESC LIMIT 1
    """)
    with get_engine().connect() as conn:
        row = conn.execute(sql, {"tf": timeframe}).fetchone()
    if row:
        return {"regime": row[0], "regime_id": row[1], "confidence": row[2], "time": row[3]}
    return {"regime": "UNKNOWN", "regime_id": -1, "confidence": 0.0, "time": None}


def save_trade(trade: dict) -> int:
    """Ouvre un trade en base. Retourne l'ID du trade."""
    sql = text("""
        INSERT INTO trades
            (open_time, symbol, direction, entry_price, stop_loss, take_profit,
             lot_size, signal_score, regime_at_entry, status, mode, broker, notes)
        VALUES
            (:open_time, :symbol, :direction, :entry_price, :stop_loss, :take_profit,
             :lot_size, :signal_score, :regime_at_entry, 'OPEN', :mode, :broker, :notes)
        RETURNING id
    """)
    with get_engine().begin() as conn:
        result = conn.execute(sql, trade)
        trade_id = result.fetchone()[0]
    logger.info(f"📋 Trade #{trade_id} ouvert : {trade['direction']} @ {trade['entry_price']}")
    return trade_id


def close_trade(trade_id: int, exit_price: float, pnl: float, pnl_pct: float):
    """Ferme un trade en base."""
    sql = text("""
        UPDATE trades
        SET close_time = NOW(), exit_price = :exit_price,
            pnl = :pnl, pnl_pct = :pnl_pct, status = 'CLOSED'
        WHERE id = :trade_id
    """)
    with get_engine().begin() as conn:
        conn.execute(sql, {"trade_id": trade_id, "exit_price": exit_price, "pnl": pnl, "pnl_pct": pnl_pct})
    logger.info(f"✅ Trade #{trade_id} fermé : P&L={pnl:+.2f}$ ({pnl_pct:+.2f}%)")


def get_daily_pnl() -> float:
    """Retourne le P&L du jour en cours."""
    sql = text("""
        SELECT COALESCE(SUM(pnl), 0)
        FROM trades
        WHERE status = 'CLOSED'
          AND close_time > CURRENT_DATE
    """)
    with get_engine().connect() as conn:
        return float(conn.execute(sql).scalar())
