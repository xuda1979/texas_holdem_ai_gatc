# Ensure src/ imports and optional deterministic seeding for local runs.
import poker_ai_bootstrap as _pab

_pab.seed_all()  # no-op unless RUN_DETERMINISTIC=1 or SEED is set
del _pab

from poker_ai.cli.play import main

if __name__ == "__main__":
    main()
