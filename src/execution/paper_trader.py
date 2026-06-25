"""
Paper Trader — Boucle principale du trading automatisé.

Architecture inspirée de :
  - Two Sigma : séparation stricte signal / risk / execution
  - Citadel    : pipeline signal → sizing → order management
  - Man AHL    : boucle temps-réel avec filtres de régime

Fonctionnement :
  1. Se connecte à MT5 (Exness Demo)
  2. Charge les dernières bougies H1 depuis la DB (ou yfinance en fallback)
  3. Calcule les features + signal LightGBM + régime HMM
  4. Si signal valide (session, score, régime) → place l'ordre
  5. Vérifie les positions ouvertes
  6. Boucle toutes les LOOP_INTERVAL secondes
  7. Envoie les alertes Telegram
"""

import time
import sys
import os
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger

from src.config import (
    SYMBOL_MT5, MIN_SIGNAL_SCORE, MAX_RISK_PER_TRADE, MAX_DAILY_DRAWDOWN,
    ATR_SL_MULTIPLIER, ATR_TP_TREND, ATR_TP_RANGE,
    KELLY_FRACTION, REGIME_CHAOS, REGIME_TREND,
    SESSION_LONDON_START, SESSION_LONDON_END,
    SESSION_NY_START, SESSION_NY_END,
)
from src.models.regime_detector import RegimeDetector
from src.models.signal_generator import SignalGenerator
from src.models.feature_builder import build_features
from src.execution.mt5_client import MT5Client
from src.notifications.telegram_bot import notify_sync
from src.data.live_pipeline import fetch_all_live

# ── Paramètres de la boucle ───────────────────────────────────────────────────
LOOP_INTERVAL   = 60 * 5       # Vérifier toutes les 5 minutes
BARS_HISTORY    = 2000          # Bougies H1 à charger pour les features
MAX_POSITIONS   = 1             # Une seule position à la fois (risk management)

# ── Kelly Paper vs Live ───────────────────────────────────────────────────────
# Pendant le paper trading, on teste Kelly 0.40 pour avoir des données réalistes
# sur ce que ça donnerait en live (au lieu de 0.25 trop conservateur)
KELLY_PAPER = 0.40


class PaperTrader:
    """
    Orchestrateur principal : signal → risk → execution.

    Phase actuelle : paper trading (demo Exness).
    Objectif : valider le système sur 4 semaines avant de passer en live.
    """

    def __init__(self):
        self.mt5        = MT5Client()
        self.regime     = RegimeDetector()
        self.signal_gen = SignalGenerator()
        self.daily_pnl  = 0.0
        self.start_balance = 0.0
        self.open_trade    = None   # dict du trade en cours ou None
        self.trade_log     = []     # historique des trades

    # ─────────────────────────────────────────────────────────
    # Données
    # ─────────────────────────────────────────────────────────

    def load_bars(self) -> pd.DataFrame | None:
        """
        Charge les dernières bougies H1 gold.
        Essaie d'abord yfinance (données temps réel), fallback DB si nécessaire.
        """
        try:
            df = yf.download("GC=F", period="120d", interval="1h",
                             progress=False, auto_adjust=True)
            if df.empty:
                logger.warning("yfinance GC=F vide — pas de données")
                return None

            if hasattr(df.columns, 'get_level_values'):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index, utc=True)
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            logger.debug(f"Données chargées : {len(df)} bougies jusqu'à {df.index[-1]}")
            return df

        except Exception as e:
            logger.error(f"Erreur chargement données : {e}")
            return None

    # ─────────────────────────────────────────────────────────
    # Filtre de session
    # ─────────────────────────────────────────────────────────

    def in_active_session(self) -> bool:
        """Vérifie qu'on est dans la session London ou NY (GMT)."""
        hour = datetime.now(timezone.utc).hour
        london = SESSION_LONDON_START <= hour < SESSION_LONDON_END
        ny     = SESSION_NY_START     <= hour < SESSION_NY_END
        return london or ny

    # ─────────────────────────────────────────────────────────
    # Calcul SL / TP à partir de l'ATR
    # ─────────────────────────────────────────────────────────

    def calc_sl_tp(
        self,
        direction: str,
        entry_price: float,
        atr: float,
        regime_id: int,
    ) -> tuple[float, float]:
        """
        SL  = ATR × 2 (fixe)
        TP  = ATR × 5 en TREND, ATR × 2 en RANGE
        Basé sur Triple Barrier Method (Lopez de Prado, 2018).
        """
        sl_dist = atr * ATR_SL_MULTIPLIER
        tp_mult = ATR_TP_TREND if regime_id == REGIME_TREND else ATR_TP_RANGE
        tp_dist = atr * tp_mult

        if direction == "LONG":
            sl = round(entry_price - sl_dist, 2)
            tp = round(entry_price + tp_dist, 2)
        else:
            sl = round(entry_price + sl_dist, 2)
            tp = round(entry_price - tp_dist, 2)

        return sl, tp

    # ─────────────────────────────────────────────────────────
    # Vérification drawdown journalier
    # ─────────────────────────────────────────────────────────

    def check_daily_limit(self) -> bool:
        """Stoppe le trading si on dépasse la perte journalière max (5%)."""
        if self.start_balance <= 0:
            return True
        pnl_pct = self.daily_pnl / self.start_balance
        if pnl_pct <= -MAX_DAILY_DRAWDOWN:
            logger.warning(
                f"⛔ Limite journalière atteinte : {pnl_pct*100:.1f}% "
                f"(max autorisé : {-MAX_DAILY_DRAWDOWN*100:.0f}%)"
            )
            return False
        return True

    # ─────────────────────────────────────────────────────────
    # Vérification des positions MT5
    # ─────────────────────────────────────────────────────────

    def sync_positions(self):
        """
        Synchronise l'état local avec MT5.
        Si notre trade_en_cours a été fermé par SL/TP → met à jour le PnL.
        Ferme de force après TIME_STOP_HOURS si ni TP ni SL touché.
        """
        if self.open_trade is None:
            return

        ticket = self.open_trade["ticket"]

        # ── Time stop (24h max) ───────────────────────────────────────────────
        TIME_STOP_HOURS = 24
        try:
            raw_time = self.open_trade.get("time", "")
            # Deux formats possibles : timestamp MT5 (int) ou isoformat (str bot)
            if str(raw_time).lstrip("-").isdigit():
                trade_open = datetime.fromtimestamp(int(raw_time), tz=timezone.utc)
            else:
                trade_open = datetime.fromisoformat(str(raw_time))
                if trade_open.tzinfo is None:
                    trade_open = trade_open.replace(tzinfo=timezone.utc)

            hours_open = (datetime.now(timezone.utc) - trade_open).total_seconds() / 3600
            if hours_open >= TIME_STOP_HOURS:
                direction = self.open_trade.get("direction", "?")
                logger.warning(
                    f"⏱ Time stop — {direction} ouvert depuis {hours_open:.1f}h → fermeture forcée"
                )
                notify_sync(
                    f"⏱ *Time Stop déclenché*\n"
                    f"Direction : `{direction}`\n"
                    f"Durée : `{hours_open:.0f}h` (max {TIME_STOP_HOURS}h)\n"
                    f"Fermeture forcée du ticket `{ticket}`."
                )
                self.mt5.close_trade(ticket)
        except Exception as e:
            logger.debug(f"Time stop check : {e}")
        # ─────────────────────────────────────────────────────────────────────

        positions = self.mt5.get_positions()
        still_open = any(p["ticket"] == ticket for p in positions)

        if not still_open:
            import MetaTrader5 as mt5_lib
            deals = mt5_lib.history_deals_get(position=ticket)
            if deals:
                pnl = sum(d.profit for d in deals)
                self.daily_pnl += pnl
                result = "✅ TP" if pnl > 0 else "❌ SL"
                logger.info(f"{result} touché — Ticket {ticket} | PnL: {pnl:+.2f}$")
                direction = self.open_trade.get("direction", "?")
                entry     = self.open_trade.get("entry", 0)
                icon      = "✅" if pnl > 0 else "❌"
                notify_sync(
                    f"{icon} *Trade fermé* — {direction}\n"
                    f"Entrée : `{entry:.2f}$`\n"
                    f"PnL : `{pnl:+.2f}$`\n"
                    f"PnL jour : `{self.daily_pnl:+.2f}$`"
                )
            self.open_trade = None

    # ─────────────────────────────────────────────────────────
    # Boucle principale
    # ─────────────────────────────────────────────────────────

    def run_once(self):
        """Un cycle complet : données → signal → ordre."""

        # 1. Synchroniser les positions existantes
        self.sync_positions()

        # 2. Vérifier le drawdown journalier
        if not self.check_daily_limit():
            notify_sync(
                f"⛔ *Limite journalière atteinte*\n"
                f"PnL jour : `{self.daily_pnl:+.2f}$`\n"
                f"Trading suspendu jusqu'à demain."
            )
            return

        # 3. Vérifier la session active
        if not self.in_active_session():
            logger.debug("⏰ Hors session (London/NY) — pas de signal")
            return

        # 4. Position déjà ouverte → ne pas ouvrir une deuxième
        if self.open_trade is not None:
            bid, ask = self.mt5.get_current_price()
            logger.debug(
                f"📊 Position ouverte : {self.open_trade['direction']} "
                f"@ {self.open_trade['entry']:.2f} | Prix actuel {bid:.2f}"
            )
            return

        # 5. Charger les données (OHLCV + COT + FRED + ETF + Sentiment)
        df = fetch_all_live()
        if df is None or len(df) < 200:
            logger.warning("Données insuffisantes pour générer un signal")
            return

        # 6. Régime HMM
        try:
            _, regime_id, regime_conf = self.regime.predict(df)
        except Exception as e:
            logger.error(f"Erreur régime HMM : {e}")
            return

        if regime_id == REGIME_CHAOS:
            logger.debug(f"🌀 Régime CHAOS ({regime_conf:.0%}) — pas de trade")
            return

        if regime_id == REGIME_TREND:
            logger.debug(f"📈 Régime TREND ({regime_conf:.0%}) — modèle sans edge (WR 34.3%), trade ignoré")
            return

        # 7. Signal LightGBM
        try:
            features = build_features(df, cot_df=None, macro_dict={},
                                      regime_id=regime_id, regime_confidence=regime_conf)
            score, direction = self.signal_gen.predict(features.iloc[[-1]])
        except Exception as e:
            logger.error(f"Erreur signal LightGBM : {e}")
            return

        regime_name = {REGIME_CHAOS: "CHAOS", REGIME_TREND: "TREND"}.get(regime_id, "RANGE")

        logger.info(
            f"📡 Signal : {direction} | Score {score:.1f} | "
            f"Régime {regime_name} ({regime_conf:.0%})"
        )

        # Notification Telegram pour tout signal non-FLAT
        if direction != "FLAT":
            bid, _ = self.mt5.get_current_price()
            emoji  = "🟢" if direction == "LONG" else "🔴"
            notify_sync(
                f"{emoji} *Signal GoldBot*\n"
                f"Direction : `{direction}`\n"
                f"Score : `{score:.1f}/100`\n"
                f"Régime : `{regime_name}`\n"
                f"Prix : `{bid:.2f}$`"
            )

        # 8. Filtrer les signaux faibles
        if direction == "FLAT" or score < MIN_SIGNAL_SCORE:
            logger.debug(f"Signal FLAT ou score trop bas ({score:.1f} < {MIN_SIGNAL_SCORE})")
            return

        # 9. Filtre NEXUS — validation institutionnelle multi-couches
        try:
            from src.models.nexus_features import compute_nexus_features
            df_nexus       = compute_nexus_features(df)
            nexus_direction = float(df_nexus["nexus_direction"].iloc[-1])
            nexus_global    = float(df_nexus["nexus_global_score"].iloc[-1])

            NEXUS_THRESHOLD = 0.30   # ±30 sur l'échelle -1/+1 = score global < 35 ou > 65

            long_blocked  = direction == "LONG"  and nexus_direction < -NEXUS_THRESHOLD
            short_blocked = direction == "SHORT" and nexus_direction >  NEXUS_THRESHOLD

            logger.info(
                f"🧭 NEXUS : global={nexus_global:.0f} | direction={nexus_direction:+.2f} "
                f"| signal={direction} → {'⛔ BLOQUÉ' if long_blocked or short_blocked else '✅ OK'}"
            )

            if long_blocked or short_blocked:
                notify_sync(
                    f"🧭 *Signal bloqué par NEXUS*\n"
                    f"Signal LightGBM : `{direction}` (score {score:.0f})\n"
                    f"Score NEXUS : `{nexus_global:.0f}/100`\n"
                    f"Contradiction macro — trade annulé."
                )
                return
        except Exception as e:
            logger.warning(f"⚠️ NEXUS filter : {e} — signal non filtré")

        # 10. Calculer l'ATR pour SL/TP
        atr = df["high"].tail(20).values - df["low"].tail(20).values
        atr_val = float(np.mean(atr))

        # 11. Prix actuel
        bid, ask = self.mt5.get_current_price()
        if bid == 0:
            return

        entry   = ask if direction == "LONG" else bid
        sl, tp  = self.calc_sl_tp(direction, entry, atr_val, regime_id)
        sl_dist = abs(entry - sl)

        # 12. Taille de position — Kelly dynamique selon conviction du signal
        # Plus le score est élevé, plus on size la position
        if score >= 85:
            kelly = 0.70   # conviction maximale
        elif score >= 75:
            kelly = 0.55   # signal fort
        elif score >= 65:
            kelly = 0.35   # signal normal
        else:
            kelly = 0.20   # signal faible, position minimale

        balance  = self.mt5.get_balance()
        lot_size = self.mt5.calc_lot_size(
            capital=balance,
            risk_pct=MAX_RISK_PER_TRADE,
            sl_distance_usd=sl_dist,
            kelly=kelly,
        )
        logger.debug(f"  Kelly dynamique : score={score:.0f} → kelly={kelly} → {lot_size} lot")

        # 13. Placer l'ordre
        logger.info(
            f"🎯 Ordre : {direction} | {lot_size} lot | Entry~{entry:.2f} "
            f"| SL={sl:.2f} | TP={tp:.2f} | ATR={atr_val:.2f}"
        )

        result = self.mt5.open_trade(
            direction=direction,
            lot_size=lot_size,
            sl_price=sl,
            tp_price=tp,
            comment=f"GoldBot {direction} s{score:.0f}",
        )

        if result:
            self.open_trade = result
            self.trade_log.append({
                **result,
                "score":      score,
                "regime":     regime_name,
                "atr":        atr_val,
            })
            emoji = "🟢" if direction == "LONG" else "🔴"
            notify_sync(
                f"{emoji} *Trade ouvert*\n"
                f"Direction : `{direction}`\n"
                f"Entrée : `{entry:.2f}$`\n"
                f"SL : `{sl:.2f}$`\n"
                f"TP : `{tp:.2f}$`\n"
                f"Lots : `{lot_size}`\n"
                f"Score : `{score:.1f}/100` | Régime : `{regime_name}`"
            )

    def run(self):
        """Boucle infinie — tourne jusqu'à Ctrl+C."""
        logger.info("=" * 55)
        logger.info("  🤖 PAPER TRADER — GoldBot XAU/USD")
        logger.info(f"  Kelly paper : {KELLY_PAPER} | Score min : {MIN_SIGNAL_SCORE}")
        logger.info(f"  Boucle toutes les {LOOP_INTERVAL//60} min")
        logger.info("=" * 55)

        if not self.mt5.connect():
            logger.error("Impossible de se connecter à MT5 — arrêt")
            sys.exit(1)

        self.start_balance = self.mt5.get_balance()
        logger.info(f"  Balance de départ : {self.start_balance:.2f} USD")

        # Vérifier si une position est déjà ouverte dans MT5 au démarrage
        existing = self.mt5.get_positions()
        if existing:
            pos = existing[0]
            self.open_trade = {
                "ticket":    pos["ticket"],
                "direction": "LONG" if pos["type"] == 0 else "SHORT",
                "lot_size":  pos["volume"],
                "entry":     pos["price_open"],
                "sl":        pos["sl"],
                "tp":        pos["tp"],
                "time":      str(pos["time"]),
            }
            logger.warning(f"⚠️ Position existante détectée : {self.open_trade['direction']} {self.open_trade['lot_size']} lot @ {self.open_trade['entry']:.2f} — pas de nouveau trade")

        notify_sync(
            f"🤖 *GoldBot démarré*\n"
            f"Mode : `PAPER TRADING`\n"
            f"Balance : `{self.start_balance:.2f}$`\n"
            f"Score min : `{MIN_SIGNAL_SCORE}`\n"
            f"Kelly : `{KELLY_PAPER}`\n"
            f"Compte : Exness Demo `436521787`"
        )

        try:
            while True:
                try:
                    self.run_once()
                except Exception as e:
                    logger.error(f"Erreur dans le cycle : {e}")

                time.sleep(LOOP_INTERVAL)

        except KeyboardInterrupt:
            logger.info("\n⏹ Arrêt demandé — fermeture des positions...")
            # Ne ferme PAS automatiquement — laisser SL/TP gérer
            self.mt5.disconnect()
            logger.info(f"  Trades effectués : {len(self.trade_log)}")
            logger.info(f"  PnL journalier   : {self.daily_pnl:+.2f}$")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    logger.add("logs/paper_trader.log", level="DEBUG", rotation="10 MB", retention="30 days")

    trader = PaperTrader()
    trader.run()
