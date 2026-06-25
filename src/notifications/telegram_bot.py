"""
Telegram Bot — Interface de contrôle du GoldBot.
Groupe : Trading Ai System | Bot : @fire_goldbot

Sources :
  - python-telegram-bot v21 (PTB team, international)
    Documentation officielle : https://python-telegram-bot.org/
  - Telegram Bot API (Telegram, Durov brothers — Russie/UAE)
  - Architecture inspirée des bots de trading institutionnels :
    Two Sigma (USA), Citadel (USA) utilisent des interfaces chat internes
    pour monitorer leurs systèmes en temps réel

Commandes disponibles :
  /status   → État complet du système
  /signal   → Signal LightGBM actuel
  /regime   → Régime HMM actuel (TREND/RANGE/CHAOS)
  /pause    → Suspendre les trades (garde les positions)
  /resume   → Reprendre les trades
  /report   → Rapport de performance
  /help     → Liste des commandes
"""

import asyncio
from datetime import datetime
from loguru import logger

from telegram import Update, Bot
from telegram.ext import (
    Application, CommandHandler, ContextTypes, MessageHandler, filters
)
from telegram.constants import ParseMode

from src.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, TRADING_MODE


# ─────────────────────────────────────────────
# État global du bot (simple, pas de DB)
# ─────────────────────────────────────────────

class BotState:
    trading_paused: bool = False
    pause_reason: str = ""
    start_time: datetime = datetime.now()

state = BotState()


# ─────────────────────────────────────────────
# Helpers de formatage
# ─────────────────────────────────────────────

def _regime_emoji(regime: str) -> str:
    return {"TREND": "📈", "RANGE": "↔️", "CHAOS": "⚠️"}.get(regime, "❓")

def _direction_emoji(direction: str) -> str:
    return {"LONG": "🟢", "SHORT": "🔴", "FLAT": "⬜"}.get(direction, "❓")

def _score_bar(score: float) -> str:
    filled = int(score / 10)
    return "█" * filled + "░" * (10 - filled)


# ─────────────────────────────────────────────
# Commandes
# ─────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🏆 *GoldBot — Trading XAU/USD*\n\n"
        "Bienvenue dans le groupe *Trading Ai System*\!\n\n"
        "Tape /help pour voir les commandes disponibles\.",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📋 *Commandes disponibles*\n\n"
        "/status  — État complet du système\n"
        "/signal  — Signal actuel \(score 0\-100\)\n"
        "/regime  — Régime de marché HMM\n"
        "/pause   — Suspendre les trades\n"
        "/resume  — Reprendre les trades\n"
        "/report  — Rapport de performance\n"
        "/help    — Cette aide"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche l'état complet du système."""
    try:
        from src.models.regime_detector import get_current_regime
        from src.models.signal_generator import get_current_signal
        from src.database import get_daily_pnl

        regime_name, regime_conf = get_current_regime("1h")
        score, direction = get_current_signal("1h")
        daily_pnl = get_daily_pnl()

        mode_icon = "📄" if TRADING_MODE == "paper" else "💰"
        pause_line = "⏸️ *TRADING EN PAUSE*\n" if state.trading_paused else ""
        uptime = str(datetime.now() - state.start_time).split(".")[0]

        text = (
            f"{pause_line}"
            f"⚡ *GoldBot — Statut*\n\n"
            f"{mode_icon} Mode : `{TRADING_MODE.upper()}`\n"
            f"⏱️ Uptime : `{uptime}`\n\n"
            f"*Marché*\n"
            f"{_regime_emoji(regime_name)} Régime : `{regime_name}` \({regime_conf:.0%}\)\n"
            f"{_direction_emoji(direction)} Signal : `{direction}` — `{score:.1f}/100`\n"
            f"`{_score_bar(score)}`\n\n"
            f"*Performance aujourd'hui*\n"
            f"{'🟢' if daily_pnl >= 0 else '🔴'} P&L : `{daily_pnl:+.2f}$`\n"
        )

        if state.trading_paused:
            text += f"\n⏸️ Pause : _{state.pause_reason}_"

    except Exception as e:
        text = f"❌ Erreur statut : `{e}`"

    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche le signal LightGBM actuel."""
    try:
        from src.models.signal_generator import get_current_signal
        score, direction = get_current_signal("1h")

        emoji = _direction_emoji(direction)
        bar   = _score_bar(score)
        min_score = 70

        if score >= min_score:
            verdict = "✅ Signal suffisant pour trader"
        elif score >= 55:
            verdict = "⏳ Signal faible — en attente"
        else:
            verdict = "❌ Signal insuffisant"

        text = (
            f"{emoji} *Signal LightGBM*\n\n"
            f"Direction : `{direction}`\n"
            f"Score : `{score:.1f}/100`\n"
            f"`{bar}`\n\n"
            f"Seuil minimum : `{min_score}/100`\n"
            f"{verdict}"
        )

    except Exception as e:
        text = f"❌ Erreur signal : `{e}`"

    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_regime(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche le régime HMM actuel."""
    try:
        from src.models.regime_detector import get_current_regime

        regime_name, confidence = get_current_regime("1h")
        emoji = _regime_emoji(regime_name)

        descriptions = {
            "TREND": "Tendance directionnelle forte\. Le bot est actif\.",
            "RANGE": "Marché en consolidation\. Signaux plus rares\.",
            "CHAOS": "Volatilité extrême \(news, événement macro\)\. *Trading suspendu automatiquement*\.",
        }

        text = (
            f"{emoji} *Régime HMM actuel*\n\n"
            f"Régime : `{regime_name}`\n"
            f"Confiance : `{confidence:.1%}`\n\n"
            f"_{descriptions.get(regime_name, '')}_"
        )

    except Exception as e:
        text = f"❌ Erreur régime : `{e}`"

    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Suspend les trades."""
    reason = " ".join(context.args) if context.args else "Pause manuelle"
    state.trading_paused = True
    state.pause_reason   = reason

    text = (
        f"⏸️ *Trading suspendu*\n\n"
        f"Raison : _{reason}_\n\n"
        f"Les positions ouvertes sont conservées\.\n"
        f"Tape /resume pour reprendre\."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)
    logger.warning(f"Trading suspendu via Telegram — {reason}")


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reprend les trades."""
    state.trading_paused = False
    state.pause_reason   = ""

    text = (
        "▶️ *Trading repris*\n\n"
        "Le bot surveille à nouveau le marché\.\n"
        "Prochain signal dans la prochaine bougie\."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)
    logger.info("Trading repris via Telegram")


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Rapport de performance."""
    try:
        from src.database import get_daily_pnl, get_engine
        from sqlalchemy import text as sql_text

        daily_pnl = get_daily_pnl()

        with get_engine().connect() as conn:
            row = conn.execute(sql_text("""
                SELECT
                    COUNT(*) FILTER (WHERE status='CLOSED') as total,
                    COUNT(*) FILTER (WHERE status='CLOSED' AND pnl > 0) as wins,
                    COUNT(*) FILTER (WHERE status='OPEN') as open_trades,
                    COALESCE(SUM(pnl) FILTER (WHERE status='CLOSED'), 0) as total_pnl
                FROM trades
                WHERE mode = :mode
            """), {"mode": TRADING_MODE}).fetchone()

        total      = row[0] or 0
        wins       = row[1] or 0
        open_count = row[2] or 0
        total_pnl  = row[3] or 0.0
        winrate    = (wins / total * 100) if total > 0 else 0.0

        def esc(v: str) -> str:
            for c in r"\.+-()":
                v = v.replace(c, f"\\{c}")
            return v

        text = (
            f"📊 *Rapport de performance*\n"
            f"Mode : `{TRADING_MODE.upper()}`\n\n"
            f"*Trades fermés*\n"
            f"Total : `{total}`\n"
            f"Gagnants : `{wins}` \\({esc(f'{winrate:.1f}')}%\\)\n"
            f"P&L total : `{esc(f'{total_pnl:+.2f}')}$`\n"
            f"P&L aujourd'hui : `{esc(f'{daily_pnl:+.2f}')}$`\n\n"
            f"*Positions ouvertes*\n"
            f"En cours : `{open_count}`"
        )

    except Exception as e:
        text = f"❌ Erreur rapport : `{e}`"

    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


# ─────────────────────────────────────────────
# Envoi de notifications (appelé par le bot)
# ─────────────────────────────────────────────

async def send_signal_alert(score: float, direction: str, regime: str, price: float):
    """Envoie une alerte signal dans le groupe Trading Ai System."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    emoji = _direction_emoji(direction)
    bar   = _score_bar(score)

    text = (
        f"{emoji} *Nouveau Signal Gold*\n\n"
        f"Direction : `{direction}`\n"
        f"Score : `{score:.1f}/100`\n"
        f"`{bar}`\n\n"
        f"Prix actuel : `{price:.2f}$`\n"
        f"Régime : `{regime}` {_regime_emoji(regime)}\n\n"
        f"_Signal généré automatiquement par GoldBot_"
    )

    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=text,
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    except Exception as e:
        logger.error(f"Erreur envoi Telegram : {e}")


async def send_trade_notification(action: str, direction: str, price: float,
                                   sl: float, tp: float, lot: float, score: float):
    """Notifie l'ouverture ou fermeture d'un trade."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    if action == "OPEN":
        emoji = "🟢" if direction == "LONG" else "🔴"
        text = (
            f"{emoji} *Trade Ouvert*\n\n"
            f"Direction : `{direction}`\n"
            f"Entrée : `{price:.2f}$`\n"
            f"Stop Loss : `{sl:.2f}$`\n"
            f"Take Profit : `{tp:.2f}$`\n"
            f"Lot size : `{lot:.4f}`\n"
            f"Score signal : `{score:.1f}/100`"
        )
    else:
        pnl = price - sl if direction == "LONG" else sl - price
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        text = (
            f"{pnl_emoji} *Trade Fermé*\n\n"
            f"Direction : `{direction}`\n"
            f"Prix de sortie : `{price:.2f}$`\n"
            f"P&L : `{pnl:+.2f}$`"
        )

    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=text,
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    except Exception as e:
        logger.error(f"Erreur envoi Telegram : {e}")


def notify_sync(text: str):
    """Envoi synchrone simple pour les alertes d'urgence."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
        }, timeout=10)
    except Exception as e:
        logger.error(f"notify_sync erreur : {e}")


# ─────────────────────────────────────────────
# Démarrage du bot
# ─────────────────────────────────────────────

def run_bot():
    """Lance le bot Telegram en mode polling."""
    if not TELEGRAM_TOKEN:
        logger.error("❌ TELEGRAM_TOKEN manquant dans .env")
        return

    logger.info(f"🤖 Démarrage @fire_goldbot — groupe : Trading Ai System")

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("regime", cmd_regime))
    app.add_handler(CommandHandler("pause",  cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("report", cmd_report))

    logger.success("✅ Bot actif — en attente de commandes dans Trading Ai System")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_bot()
