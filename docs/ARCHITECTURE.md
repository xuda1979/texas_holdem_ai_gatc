# Modular Architecture (Separation Principle)

This repository now includes a lightweight modular layer under `src/gatc_modular/`
that cleanly separates responsibilities and enables independent development and
testing of subsystems:

```
Engine (game rules, state, transitions)
    ↑           (ports/engine.py: Protocols)
Policy (agent/AI choosing an action)
    ↑           (ports/policy.py: Protocols)
GameLoop (turn-taking, legality checks, rewards aggregation)
    ↑           (services/game_loop.py)
SelfPlayService (episodes orchestration, stats)
                (services/self_play.py)
```

## Why this layer?
* **Parallel work**: engine devs, policy/AI devs, GUI devs, and trainer devs can work independently.
* **Testability**: pure, typed protocols allow fast unit tests with small fakes (see `gatc_modular/testing`).
* **Replaceability**: swap engines (CFR, RL envs, GUI-driven engine) or policies (random, neural, CFR) with no changes to services.
* **Safety**: illegal actions are detected or auto-corrected in one place (the `GameLoop`).

## Key Contracts
### Engine Port
`ports/engine.py` defines a `Protocol` that any engine must satisfy:
* `num_players` – int
* `reset(seed: Optional[int]) -> Observation`
* `current_player() -> PlayerId`
* `legal_actions() -> List[Action]`
* `step(action: Action) -> StepResult`
* `is_terminal() -> bool`
* `winner() -> Optional[PlayerId]`
* `clone() -> Engine`

### Policy Port
`ports/policy.py` defines a `Policy` with:
* `select_action(obs, legal_actions, player_id) -> Action`

### Trainer Port (optional)
`ports/trainer.py` sketches a general trainer interface so training code can be plugged in later.

## Services
### `GameLoop`
* Orchestrates a single episode.
* Validates and (optionally) auto-corrects illegal actions.
* Aggregates rewards.

### `SelfPlayService`
* Runs many episodes and returns aggregate statistics (wins, total rewards, avg steps).

## Testing utilities
* `testing/fakes.py` includes a tiny deterministic 2-player engine (`CountingGameEngine`) and simple policies.
* Unit tests under `tests/modular/` demonstrate the contracts and orchestration logic.

## Integrating existing code
You can adapt your current engine/GUI/trainers in small steps:
1. Make your engine object satisfy the `Engine` Protocol (or write an adapter class).
2. Ensure your policy/agent object implements `select_action(obs, legal_actions, player_id)`.
3. Use `GameLoop` in CLI/GUI code to run episodes without duplicating turn logic.
4. Use `SelfPlayService` to collect stats for evaluation.

This keeps domain code (poker rules/CFR/RL) decoupled from orchestration and presentation.
