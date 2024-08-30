from gui import PokerGameGUI
from ai_cfr_trainer import AICFRTrainer

def main():
    trainer = AICFRTrainer()
    PokerGameGUI(trainer)

if __name__ == "__main__":
    main()
