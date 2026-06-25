"""
Point d'entrée principal du GoldBot.
Lance la collecte initiale puis attend les prochaines étapes.
"""
from loguru import logger
import sys

from src.config import TRADING_MODE, DB_URL
from src.database import test_connection
from src.data.collector import run_initial_collection


def main():
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    logger.add("logs/goldbot.log", level="DEBUG", rotation="10 MB", retention="30 days")

    logger.info("=" * 55)
    logger.info("  🏆 GOLDBOT — Système de Trading XAU/USD")
    logger.info(f"  Mode : {TRADING_MODE.upper()}")
    logger.info("=" * 55)

    # Étape 1 — Vérifier la connexion base de données
    if not test_connection():
        logger.error("Impossible de continuer sans base de données. Lance d'abord : docker-compose up -d")
        sys.exit(1)

    # Étape 2 — Collecte initiale des données
    logger.info("📡 Démarrage de la collecte initiale...")
    run_initial_collection()

    logger.success("✅ Infrastructure prête. Prochaine étape : détection de régime HMM")


if __name__ == "__main__":
    main()
