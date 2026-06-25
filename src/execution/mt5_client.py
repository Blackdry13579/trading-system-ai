"""
MT5 Client — Connexion directe MetaTrader5 (package officiel MetaQuotes).

Sources :
  - MetaQuotes official Python package (MetaTrader5 v5.0.5640)
  - Exness Demo : Exness-MT5Trial9 / login 436521787
  - Inspiration architecture : Two Sigma / Citadel execution layer design

Avantage vs MetaAPI :
  - 100% gratuit, aucune limite
  - Latence locale (pas de cloud)
  - MT5 doit être ouvert sur Windows pendant le trading
"""

import os
import time
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

MT5_LOGIN    = int(os.getenv("MT5_LOGIN", "436521787"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER   = os.getenv("MT5_SERVER", "Exness-MT5Trial9")
SYMBOL       = "XAUUSDm"   # Exness Demo utilise XAUUSDm (le "m" = micro)


class MT5Client:
    """Client MT5 pour paper trading sur Exness Demo."""

    def __init__(self):
        self.connected  = False
        self.account_info = None

    # ─────────────────────────────────────────────
    # Connexion / Déconnexion
    # ─────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialise MT5 et se connecte au compte Exness Demo."""
        if not mt5.initialize():
            logger.error(f"❌ MT5 initialize() échoué : {mt5.last_error()}")
            logger.error("   → Assure-toi que MetaTrader5 est ouvert sur la machine")
            return False

        if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            logger.error(f"❌ Login MT5 échoué : {mt5.last_error()}")
            mt5.shutdown()
            return False

        self.account_info = mt5.account_info()
        self.connected    = True

        logger.success(f"✅ MT5 connecté — {MT5_SERVER}")
        logger.info(f"   Login     : {self.account_info.login}")
        logger.info(f"   Balance   : {self.account_info.balance:.2f} {self.account_info.currency}")
        logger.info(f"   Leverage  : 1:{self.account_info.leverage}")
        logger.info(f"   Mode      : {'Demo' if self.account_info.trade_mode == 0 else 'Live'}")
        return True

    def disconnect(self):
        mt5.shutdown()
        self.connected = False
        logger.info("MT5 déconnecté")

    # ─────────────────────────────────────────────
    # Informations du compte
    # ─────────────────────────────────────────────

    def get_balance(self) -> float:
        info = mt5.account_info()
        return info.balance if info else 0.0

    def get_equity(self) -> float:
        info = mt5.account_info()
        return info.equity if info else 0.0

    def get_positions(self) -> list[dict]:
        """Retourne toutes les positions ouvertes sur XAUUSD."""
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions is None:
            return []
        return [p._asdict() for p in positions]

    # ─────────────────────────────────────────────
    # Calcul du lot size (Kelly × ATR)
    # ─────────────────────────────────────────────

    def calc_lot_size(
        self,
        capital: float,
        risk_pct: float,
        sl_distance_usd: float,
        kelly: float = 0.25,
    ) -> float:
        """
        Calcule le lot size selon Kelly fractionnelle.

        Utilise trade_tick_value / trade_tick_size pour obtenir le vrai
        multiplicateur PnL par lot — seule méthode fiable sur MT5 quel
        que soit le symbole (XAUUSDm micro ou XAUUSD standard).

        Formule :
            usd_per_lot  = tick_value / tick_size   ($ par lot par $ de mouvement)
            risk_amount  = capital × risk_pct × kelly
            lot          = risk_amount / (sl_distance_usd × usd_per_lot)
        """
        if sl_distance_usd <= 0:
            return 0.01

        info = mt5.symbol_info(SYMBOL)
        if info is None:
            logger.warning("symbol_info introuvable — lot size = 0.01 (sécurité)")
            return 0.01

        # Multiplicateur réel : combien de $ gagne-t-on par lot pour $1 de mouvement
        usd_per_lot = info.trade_tick_value / info.trade_tick_size
        logger.debug(f"  {SYMBOL} : tick_value={info.trade_tick_value} "
                     f"tick_size={info.trade_tick_size} → {usd_per_lot:.2f} $/lot/$")

        risk_amount  = capital * risk_pct * kelly
        risk_per_lot = sl_distance_usd * usd_per_lot

        lot = risk_amount / risk_per_lot
        lot = round(lot, 2)
        lot = max(0.01, min(lot, 2.0))   # cap absolu à 2 lots (démo $10k)
        return lot

    # ─────────────────────────────────────────────
    # Prix actuel
    # ─────────────────────────────────────────────

    def get_current_price(self) -> tuple[float, float]:
        """Retourne (bid, ask) actuel sur XAUUSD."""
        mt5.symbol_select(SYMBOL, True)   # force l'ajout dans la Market Watch
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            logger.warning(f"⚠️ Impossible de lire le prix de {SYMBOL}")
            return 0.0, 0.0
        return tick.bid, tick.ask

    # ─────────────────────────────────────────────
    # Placement d'ordres
    # ─────────────────────────────────────────────

    def open_trade(
        self,
        direction: str,
        lot_size: float,
        sl_price: float,
        tp_price: float,
        comment: str = "GoldBot",
    ) -> dict | None:
        """
        Ouvre un trade LONG ou SHORT sur XAUUSDm.

        Args:
            direction : "LONG" ou "SHORT"
            lot_size  : calculé via calc_lot_size()
            sl_price  : prix du Stop Loss
            tp_price  : prix du Take Profit
            comment   : commentaire visible dans MT5

        Returns:
            dict avec les infos du trade ou None si erreur
        """
        bid, ask = self.get_current_price()
        if bid == 0:
            return None

        if direction == "LONG":
            order_type  = mt5.ORDER_TYPE_BUY
            price       = ask   # on achète au ask
        else:
            order_type  = mt5.ORDER_TYPE_SELL
            price       = bid   # on vend au bid

        request = {
            "action":        mt5.TRADE_ACTION_DEAL,
            "symbol":        SYMBOL,
            "volume":        lot_size,
            "type":          order_type,
            "price":         price,
            "sl":            sl_price,
            "tp":            tp_price,
            "deviation":     20,          # tolérance slippage en points
            "magic":         20260101,    # identifiant du bot
            "comment":       comment,
            "type_time":     mt5.ORDER_TIME_GTC,
            "type_filling":  mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            logger.error(f"❌ Ordre {direction} échoué — retcode={code} : {mt5.last_error()}")
            return None

        logger.success(
            f"✅ Trade ouvert : {direction} {lot_size} lot @ {price:.2f} "
            f"| SL={sl_price:.2f} TP={tp_price:.2f} | ticket={result.order}"
        )
        return {
            "ticket":    result.order,
            "direction": direction,
            "lot_size":  lot_size,
            "entry":     price,
            "sl":        sl_price,
            "tp":        tp_price,
            "time":      datetime.now().isoformat(),
        }

    def close_trade(self, ticket: int) -> bool:
        """Ferme un trade par son numéro de ticket."""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.warning(f"Position {ticket} introuvable")
            return False

        pos = position[0]
        bid, ask = self.get_current_price()

        if pos.type == mt5.ORDER_TYPE_BUY:
            close_type  = mt5.ORDER_TYPE_SELL
            close_price = bid
        else:
            close_type  = mt5.ORDER_TYPE_BUY
            close_price = ask

        request = {
            "action":        mt5.TRADE_ACTION_DEAL,
            "symbol":        SYMBOL,
            "volume":        pos.volume,
            "type":          close_type,
            "position":      ticket,
            "price":         close_price,
            "deviation":     20,
            "magic":         20260101,
            "comment":       "GoldBot close",
            "type_time":     mt5.ORDER_TIME_GTC,
            "type_filling":  mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.success(f"✅ Trade {ticket} fermé")
            return True

        logger.error(f"❌ Fermeture {ticket} échouée : {mt5.last_error()}")
        return False

    def close_all(self):
        """Ferme toutes les positions ouvertes."""
        for pos in self.get_positions():
            self.close_trade(pos["ticket"])

    # ─────────────────────────────────────────────
    # Test de connexion
    # ─────────────────────────────────────────────

    def test_connection(self):
        """Vérifie la connexion et affiche les infos du compte."""
        if not self.connect():
            return

        balance   = self.get_balance()
        equity    = self.get_equity()
        bid, ask  = self.get_current_price()
        positions = self.get_positions()

        logger.info("=" * 50)
        logger.info("  🔌 TEST CONNEXION MT5 — Exness Demo")
        logger.info("=" * 50)
        logger.info(f"  Balance     : {balance:.2f} USD")
        logger.info(f"  Equity      : {equity:.2f} USD")
        logger.info(f"  XAUUSDm     : Bid={bid:.2f} / Ask={ask:.2f}")
        logger.info(f"  Positions   : {len(positions)} ouverte(s)")
        logger.info("=" * 50)

        self.disconnect()


if __name__ == "__main__":
    client = MT5Client()
    client.test_connection()
