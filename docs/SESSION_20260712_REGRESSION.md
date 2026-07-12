# Huanxin ai3 Training — 2026-07-12 Regression Investigation

## Training Status
- Job `ai3-rd-20260711T141826Z` still RUNNING on ai3.
- Progress: 416 samples (last eval) → 2838 samples now (~7× growth).
- Checkpoints saved every ~30 samples. Latest: `deep_cfr_time_2838.pth`.

## Evaluation Sweep (300 hands, seed=7)

| samples | always_fold | calling_station | random_uniform |
|---------|-------------|-----------------|----------------|
| 506     | +75.0       | +24.73          | +756.43        |
| 1002    | +75.0       | +66.17          | +870.57        |
| 1510    | +72.33      | **-115.07**     | +505.33        |
| 2003    | +74.33      | **-166.33**     | +542.40        |
| 2500    | +75.0       | **-126.40**     | +846.33        |
| 2838    | +75.0       | **-35.54**      | +540.92        |

## Finding
- Model **beats** always_fold (stable +75) and random_uniform (large positive).
- Model **regressed** on calling_station between 1002 and 1510 samples.
- Best calling_station performance was at 1002 samples (+66.17).
- The regression is consistent across 300-500 hand evaluations — not just noise.

## Hypotheses
1. **Policy-net collapse**: the average-strategy network overfits to a dominant
   action pattern that exploits random_uniform (all-in calls) but loses to
   calling_station (which calls everything).
2. **Strategy buffer contamination**: once the advantage net's regrets collapse,
   the regret-matched policy recorded into the strategy buffer becomes
   degenerate, and the policy net learns that degenerate distribution.
3. **Insufficient exploration**: `epsilon: 0.05` is configured but only applied
   in `cli/self_play.py`, NOT in `selfplay/self_play.py` used by the trainer.
   Pure regret matching provides some exploration but can collapse on
   dominated actions.

## Next Steps
- Evaluate intermediate checkpoints 1100-1500 to pin down exact regression
  sample.
- Inspect advantage_net vs policy_net weight norms in checkpoints.
- Consider: (a) adding epsilon-greedy to trainer traversal, (b) lower learning
  rate, (c) larger replay buffer warmup, (d) train policy net less frequently.
- Checkpoint 1002 may be the best model so far for calling_station.

## Eval Files (on ai3)
- `models/eval_506_300h.json`
- `models/eval_1002_300h.json`
- `models/eval_1510_300h.json`
- `models/eval_2003_300h.json`
- `models/eval_2500_300h.json`
- `models/eval_2838_300h.json` (300 hands; 500-hand eval also at `eval_2838.json`)
