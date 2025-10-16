# Texas Hold'em AI Architecture

The project is now organised around small, testable subsystems.  Each
subsystem wraps the underlying implementation and records its dependencies so
that components can be mixed and matched during experiments.

```
texas_holdem_ai_gatc/
└── src/poker_ai/
    ├── ai/                     # Model definitions and trainers
    ├── engine/                 # Core game engine and simulations
    ├── evaluation/             # Performance evaluation utilities
    ├── rules/                  # Poker rule definitions
    ├── selfplay/               # MCCFR based data generation
    ├── systems/                # New subsystem wrappers
    ├── utils/                  # Shared helpers (state representation, etc.)
    └── ...
```

## Subsystems

Each subsystem lives under `src/poker_ai/systems` and is created through a
factory method returning a `Subsystem` object.

| Subsystem | Responsibility |
|-----------|----------------|
| `CFRSubsystem` | Wraps the CFR trainers (AI, Deep, Single network). |
| `TransformerSubsystem` | Creates transformer advantage networks. |
| `EmbeddingSubsystem` | Provides helpers for converting a game state into tensors. |
| `SelfPlaySubsystem` | Drives MCCFR self-play simulations. |
| `TrainingSubsystem` | Coordinates self-play and CFR optimisation. |
| `RulesSubsystem` | Supplies poker rules and game creation helpers. |
| `LoggingSubsystem` | Centralises logging utilities. |
| `EvaluationSubsystem` | Runs evaluation tournaments and score aggregation. |

Subsystems can be registered inside a `Registry` to make configuration driven
assembly trivial:

```python
from poker_ai.systems import Registry, CFRSubsystem, SelfPlaySubsystem, TrainingSubsystem

registry = Registry()
cfr = CFRSubsystem.create(device="cpu")
self_play = SelfPlaySubsystem.create(cfr_trainer=cfr.component)
training = TrainingSubsystem.create(cfr=cfr, self_play=self_play, iterations_per_cycle=10)

for subsystem in (cfr, self_play, training):
    registry.register(subsystem)

print(registry.summary())
```

This structure keeps the original modules untouched while providing clean
integration points for future development and targeted testing.
