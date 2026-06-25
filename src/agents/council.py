"""
NEXUS Council — Système multi-agents pour décisions trading, stratégie et dev.

Inspiré des meilleures implémentations mondiales :
  - AutoGen    (Microsoft Research, USA 2023) — conversation multi-agents
    Paper: "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation"
    github.com/microsoft/autogen
  - CrewAI     (João Moura, USA 2024) — équipes d'agents spécialisés
    github.com/joaomdmoura/crewAI
  - MetaGPT    (DeepWisdom, Chine 2023) — simulation entreprise logicielle
    Paper: "MetaGPT: Meta Programming for Multi-Agent Collaborative Framework"
    github.com/geekan/MetaGPT
  - FinAgent   (UIUC, USA 2024) — agents financiers multi-modaux
    Paper: "FinAgent: A Multimodal Foundation Agent for Financial Trading"
  - Society of Mind (Marvin Minsky, MIT) — intelligence collective

Architecture :
  Chaque agent a une expertise distincte et un caractère fort.
  Ils débattent en tours successifs, challengent les autres, puis convergent.
  L'orchestrateur synthétise le consensus final.

Usage :
  python -m src.agents.council "question ou décision à débattre"
  python -m src.agents.council --filter TREND "doit-on trader en TREND ?"
"""

import os
import sys
import json
import textwrap
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
import anthropic
from loguru import logger

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL             = "claude-sonnet-4-6"   # Modèle pour tous les agents
MAX_TOKENS        = 1024
DEBATE_ROUNDS     = 2                      # Tours de débat après l'ouverture


# ── Définition des agents ─────────────────────────────────────────────────────

@dataclass
class Agent:
    name:        str
    role:        str
    emoji:       str
    system:      str
    messages:    list = field(default_factory=list)


AGENTS = [
    Agent(
        name  = "QUANT",
        role  = "Analyste Quantitatif",
        emoji = "📐",
        system = textwrap.dedent("""
            Tu es un analyste quantitatif de niveau Jane Street / Two Sigma / Renaissance Technologies.
            Tu penses en chiffres, statistiques et probabilités. Rien n'est vrai sans preuve empirique.

            Tes valeurs fondamentales :
            - L'intuition sans backtest ne vaut rien
            - Un sample size < 50 trades n'est pas significatif
            - Le Sharpe Ratio et le Profit Factor sont tes boussoles
            - Tu cites toujours tes sources (papiers académiques, résultats chiffrés)
            - Tu détectes immédiatement les biais de raisonnement (survivorship bias, overfitting, data snooping)

            Contexte du projet : GoldBot — système de trading XAU/USD avec LightGBM (56.2% WR,
            +13.67%, Sharpe 9.158, Max DD -4.86%) sur données H1. Pipeline institutionnel :
            COT CFTC, FRED macro, ETF flows, Myfxbook sentiment, HMM régimes.

            Sois direct, précis, challenge les autres agents avec des chiffres.
            Réponds en français. Maximum 200 mots par intervention.
        """).strip()
    ),

    Agent(
        name  = "RISK",
        role  = "Gestionnaire de Risque",
        emoji = "🛡️",
        system = textwrap.dedent("""
            Tu es un gestionnaire de risque senior inspiré de Bridgewater Associates et
            des desks risk de Goldman Sachs / JPMorgan.

            Tes valeurs fondamentales :
            - Protéger le capital EST la stratégie numéro 1
            - Chaque décision doit passer le test : "quel est le pire cas ?"
            - Tu penses en termes de tail risk, corrélations et drawdown
            - Le Kelly Criterion et la position sizing sont ta religion
            - Tu rappelles toujours ce qui peut mal tourner

            Contexte du projet : compte démo Exness $10,000 (actuellement ~$6,200 après perte
            due à un bug de lot size 1.97 lots). Max risk 2% par trade. HMM régimes actifs.

            Tu es le gardien — tu n'empêches pas les trades mais tu exiges que chaque risque
            soit quantifié et accepté consciemment.
            Réponds en français. Maximum 200 mots par intervention.
        """).strip()
    ),

    Agent(
        name  = "ALPHA",
        role  = "Générateur de Signal",
        emoji = "⚡",
        system = textwrap.dedent("""
            Tu es un spécialiste de la génération de signaux, inspiré de Man AHL, Winton Capital
            et AQR Capital Management. Tu penses en termes d'edge, d'information coefficient et
            de persistance des signaux.

            Tes valeurs fondamentales :
            - Un signal vaut quelque chose seulement s'il a un IC > 0.02 et ICIR > 0.40
            - La diversification des signaux réduit le risque sans réduire le retour
            - Les signaux décroissent exponentiellement (alpha decay) — timing is everything
            - Tu connais les stratégies : trend following, mean reversion, carry, value

            Contexte du projet : 14 features institutionnelles actives (COT, FRED, ETF, Myfxbook).
            Backtest révèle : edge 100% en RANGE (56.2% WR), TREND destructeur (34.3% WR).
            Signal validator implémenté avec critères IC/ICIR/AUC marginal.

            Tu proposes des améliorations concrètes et défends ou challenges les signaux existants.
            Réponds en français. Maximum 200 mots par intervention.
        """).strip()
    ),

    Agent(
        name  = "DEVIL",
        role  = "Avocat du Diable",
        emoji = "😈",
        system = textwrap.dedent("""
            Tu es l'avocat du diable — ton rôle est de trouver les failles dans chaque argument,
            de questionner les hypothèses, de forcer les autres à justifier leurs positions.
            Inspiré du concept de "Red Teaming" utilisé par les hedge funds top tier.

            Tes valeurs fondamentales :
            - Tout consensus trop rapide est suspect
            - Les backtests sont toujours biaisés — comment ?
            - "Ça a marché dans le passé" n'est pas une raison suffisante
            - Tu poses les questions inconfortables que personne ne veut poser
            - Tu joues le rôle du régulateur, du marché adverse, de la malchance

            Contexte du projet : GoldBot avec des résultats backtest qui semblent bons.
            Mais le bot vient de perdre $4,681 en live sur un seul trade bugué.
            Le gap entre backtest et réalité est ton terrain de jeu.

            Sois incisif, précis dans tes critiques, mais constructif — tu cherches la vérité.
            Réponds en français. Maximum 200 mots par intervention.
        """).strip()
    ),

    Agent(
        name  = "ARCH",
        role  = "Architecte Logiciel",
        emoji = "🏗️",
        system = textwrap.dedent("""
            Tu es un architecte logiciel senior spécialisé en systèmes financiers temps-réel.
            Inspiré des pratiques de Two Sigma, Jane Street et des meilleures équipes de fintech.

            Tes valeurs fondamentales :
            - Le code qui tourne en production doit être simple, testable, observable
            - Chaque bug en production coûte 10× plus cher qu'un bug en dev
            - La robustesse > la performance > la beauté du code
            - Monitoring, alertes et logs sont aussi importants que le code lui-même
            - Tu penses toujours à la maintenance à 6 mois

            Contexte du projet : Python 3.11, LightGBM, HMM, MT5 API, Telegram,
            PostgreSQL/TimescaleDB, Grafana. Bug critique récent : lot size 1.97 lots
            au lieu de 0.02 — causé par mauvaise lecture du contract_size MT5.

            Tu identifies les risques techniques, proposes des architectures robustes,
            et challenges les implémentations fragiles.
            Réponds en français. Maximum 200 mots par intervention.
        """).strip()
    ),
]


# ── Orchestrateur ─────────────────────────────────────────────────────────────

class NexusCouncil:
    """
    Orchestre le débat entre les agents NEXUS.

    Flow inspiré AutoGen (Microsoft) :
    1. Question posée à tous les agents
    2. Chaque agent donne sa position initiale
    3. N tours de débat : chaque agent réagit aux autres
    4. Synthèse finale par l'orchestrateur
    """

    def __init__(self, agents: list[Agent] = None):
        self.agents = agents or AGENTS
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.debate_history: list[dict] = []

    def _call_agent(self, agent: Agent, prompt: str) -> str:
        """Appelle un agent avec le contexte de la conversation."""
        messages = [{"role": "user", "content": prompt}]

        response = self.client.messages.create(
            model      = MODEL,
            max_tokens = MAX_TOKENS,
            system     = agent.system,
            messages   = messages,
        )
        return response.content[0].text.strip()

    def _format_history(self, exclude_agent: str = "") -> str:
        """Formate l'historique du débat pour le contexte d'un agent."""
        if not self.debate_history:
            return ""
        lines = ["Voici ce que les autres agents ont dit jusqu'ici:\n"]
        for entry in self.debate_history:
            if entry["agent"] != exclude_agent:
                lines.append(f"[{entry['emoji']} {entry['agent']}]: {entry['response']}\n")
        return "\n".join(lines)

    def debate(self, question: str, context: str = "") -> str:
        """
        Lance un débat complet sur une question.

        Args:
            question : la question ou décision à débattre
            context  : contexte additionnel (métriques, code, etc.)
        """
        self.debate_history = []

        print("\n" + "═" * 65)
        print(f"  🏛️  NEXUS COUNCIL — DÉBAT")
        print("═" * 65)
        print(f"\n  Question : {question}\n")

        # ── Round 0 : Position initiale de chaque agent ──────────────
        print("─" * 65)
        print("  ROUND 1 — Positions initiales")
        print("─" * 65)

        initial_prompt = f"""Question soumise au conseil :

{question}

{f"Contexte additionnel : {context}" if context else ""}

Donne ta position initiale, claire et argumentée."""

        for agent in self.agents:
            response = self._call_agent(agent, initial_prompt)
            self.debate_history.append({
                "round":    0,
                "agent":    agent.name,
                "emoji":    agent.emoji,
                "role":     agent.role,
                "response": response,
            })
            print(f"\n{agent.emoji} {agent.name} ({agent.role})")
            print("─" * 40)
            # Wrap proprement
            for line in response.split("\n"):
                if line.strip():
                    print(textwrap.fill(line, width=62, initial_indent="  ",
                                       subsequent_indent="  "))
                else:
                    print()

        # ── Rounds de débat ──────────────────────────────────────────
        for round_num in range(1, DEBATE_ROUNDS + 1):
            print(f"\n{'─'*65}")
            print(f"  ROUND {round_num + 1} — Réactions croisées")
            print("─" * 65)

            round_entries = []
            for agent in self.agents:
                history = self._format_history(exclude_agent=agent.name)
                debate_prompt = f"""Question initiale : {question}

{history}

En tenant compte des positions des autres agents, affine ou défends ta position.
Challenge un argument spécifique si tu n'es pas d'accord."""

                response = self._call_agent(agent, debate_prompt)
                round_entries.append({
                    "round":    round_num,
                    "agent":    agent.name,
                    "emoji":    agent.emoji,
                    "role":     agent.role,
                    "response": response,
                })
                print(f"\n{agent.emoji} {agent.name}")
                print("─" * 40)
                for line in response.split("\n"):
                    if line.strip():
                        print(textwrap.fill(line, width=62, initial_indent="  ",
                                           subsequent_indent="  "))
                    else:
                        print()

            self.debate_history.extend(round_entries)

        # ── Synthèse finale ───────────────────────────────────────────
        print(f"\n{'═'*65}")
        print("  🎯 SYNTHÈSE FINALE")
        print("═" * 65)

        all_opinions = self._format_history()
        synthesis_prompt = f"""Tu es l'orchestrateur du NEXUS Council.

Question débattue : {question}

Voici le débat complet entre les agents :
{all_opinions}

Produis une synthèse décisionnelle :
1. Points de consensus
2. Points de désaccord majeurs
3. Décision recommandée (avec raison)
4. Actions concrètes (max 3 bullets)

Sois direct et actionnable. 250 mots maximum."""

        synthesis_agent = Agent(
            name   = "ORCHESTRATEUR",
            role   = "Synthèse",
            emoji  = "🎯",
            system = "Tu es un orchestrateur de débat expert. Tu synthétises les positions "
                     "de multiples agents spécialisés et produis une décision claire et "
                     "actionnable. Tu es neutre, rigoureux et direct. Réponds en français.",
        )

        synthesis = self._call_agent(synthesis_agent, synthesis_prompt)
        print()
        for line in synthesis.split("\n"):
            if line.strip():
                print(textwrap.fill(line, width=62, initial_indent="  ",
                                   subsequent_indent="  "))
            else:
                print()

        print("\n" + "═" * 65)
        return synthesis

    def quick(self, question: str, agent_names: list[str] = None) -> str:
        """Débat rapide avec seulement 2-3 agents sélectionnés."""
        if agent_names:
            selected = [a for a in self.agents if a.name in agent_names]
        else:
            selected = self.agents[:3]

        council = NexusCouncil(agents=selected)
        return council.debate(question)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    logger.remove()
    logger.add(sys.stdout, level="WARNING", colorize=True,
               format="<yellow>{message}</yellow>")

    if not ANTHROPIC_API_KEY:
        print("❌ ANTHROPIC_API_KEY manquant dans .env")
        print("   Ajoute : ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    args = sys.argv[1:]

    if not args:
        print("Usage : python -m src.agents.council 'ta question'")
        print("        python -m src.agents.council --quick 'question rapide'")
        sys.exit(0)

    quick_mode = "--quick" in args
    if quick_mode:
        args = [a for a in args if a != "--quick"]

    question = " ".join(args)

    council = NexusCouncil()
    if quick_mode:
        council.quick(question, agent_names=["QUANT", "RISK", "DEVIL"])
    else:
        council.debate(question)


if __name__ == "__main__":
    main()
