"""Convenience entry point for running self-play simulations from a checkout."""

# Ensure src/ imports and optional deterministic seeding for local runs.
import poker_ai_bootstrap as _pab

_pab.seed_all()  # no-op unless RUN_DETERMINISTIC=1 or SEED is set
del _pab

# NOTE: On Windows, multiprocessing requires guard:
# if __name__ == "__main__":
#     main()
# Keep your existing entrypoint consistent with this pattern.

from poker_ai.cli.self_play import main

if __name__ == "__main__":
    main()
