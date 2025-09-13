#!/usr/bin/env python3
"""Debug script to test AICFRTrainer"""

try:
    print("1. Testing AICFRTrainer import...")
    from poker_ai.ai.trainers.ai_cfr_trainer import AICFRTrainer

    print("   ✓ AICFRTrainer imported successfully")

    print("2. Creating AICFRTrainer instance...")
    trainer = AICFRTrainer()
    print("   ✓ AICFRTrainer instance created")

    print("3. Checking trainer attributes...")
    print(f"   - Has 'model' attribute: {hasattr(trainer, 'model')}")
    print(f"   - Has 'num_actions' attribute: {hasattr(trainer, 'num_actions')}")

    if hasattr(trainer, "model"):
        print(f"   - Model type: {type(trainer.model)}")
        print(f"   - Model has 'num_actions': {hasattr(trainer.model, 'num_actions')}")
        if hasattr(trainer.model, "num_actions"):
            print(f"   - Model num_actions value: {trainer.model.num_actions}")

    print("4. All tests passed! ✓")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()
