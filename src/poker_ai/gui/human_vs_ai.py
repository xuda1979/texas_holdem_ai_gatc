from poker_ai.ai.trainers.ai_cfr_trainer import AICFRTrainer
from poker_ai.gui.gui import PokerGameGUI


def main():
    trainer = AICFRTrainer()
    PokerGameGUI(trainer)

if __name__ == "__main__":
    main()
