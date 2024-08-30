import cProfile
from ai_cfr_trainer import AICFRTrainer

def profile_training():
    profiler = cProfile.Profile()
    trainer = AICFRTrainer()
    
    profiler.enable()
    trainer.train()
    profiler.disable()
    
    profiler.print_stats()

if __name__ == "__main__":
    profile_training()
