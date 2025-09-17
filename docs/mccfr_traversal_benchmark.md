# MCCFR Traversal Performance

## Benchmark setup

* Environment: container Python runtime (same as repository CI image).
* Configuration: four-player `SelfPlay` hand with a lightweight dummy trainer
  returning zero advantages and a stub replay buffer.
* We patched `prepare_transformer_input` to return constant tensors so the
  measurement focused on traversal cost rather than feature extraction.
* Each run executed five calls to `SelfPlay.play_hand_for_training`.

## Results

| Version | Elapsed time (s) | Notes |
| ------- | ---------------- | ----- |
| Before in-place traversal | 0.522 | Baseline using state cloning per action.【687813†L1】 |
| After in-place traversal | 0.321 | Uses undo stack snapshots and reuses game objects.【1bc25e†L1】 |

The optimization reduces traversal time by approximately 39%, demonstrating
that reusing the `TexasHoldem` instance with an undo stack meaningfully lowers
the overhead of self-play simulations.
