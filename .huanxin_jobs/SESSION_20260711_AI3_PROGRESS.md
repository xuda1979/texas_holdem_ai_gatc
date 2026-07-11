# Huanxin ai3 NPU Training Session — 2026-07-11

## Environment
- **Live env**: ai3 (ASI3). ai1 (ASI1) is stopped.
- **NPU**: 4 (single NPU, 2056MB used per process)
- **Python**: 3.11.14, torch 2.9.0+cpu, torch_npu 2.9.0, treys 0.1.8
- **Project path**: `/root/work/software/texas-holdem` (NAS-mounted: `v4hw03pool04.nas.cidc-rp-210.internal`)

## Training Run: ai3-rd-20260711T141826Z
- **Config**: `poc_ai3_npu.yaml` (4-layer model, 128 hidden_dim, 32 samples/cycle, 20k max samples)
- **Algorithm**: deep_cfr
- **PID**: 1149 (worker), 1185 (launcher)
- **Log**: `/root/work/software/texas-holdem/logs/ai3-rd-20260711.log`

## Progress (loss trajectory)
| Cycle | Loss | Samples | Replay Buffer | Time |
|-------|------|---------|---------------|------|
| 1 | 4.0275 | 32 | 7001 | 14:18-14:26 (8m) |
| 2 | 1.7047 | 64 | 16042 | 14:26-14:36 (10m) |
| 3 | 1.5239 | 96 | 24932 | 14:36-14:48 (12m) |
| 4 | 1.5624 | 128 | 37497 | 14:48-15:03 (15m) |
| 5 | 1.5404 | 160 | 47419 | 15:03-15:15 (12m) |
| 6 | 1.5933 | 192 | 58990 | 15:15-15:30 (15m) |
| 7 | 1.5080 | 224 | 68375 | 15:30-15:43 (13m) |
| 8 | 1.5713 | 256 | 79491 | 15:43-15:57 (14m) |
| 9 | 1.4597 | 288 | 92368 | 15:57-16:14 (17m) |
| 10 | 1.4823 | 320 | 98856 | 16:14-16:22 (8m) |
| 11 | in progress | 320+ | — | 16:22+ |

Loss stable around 1.45-1.55. Best so far 1.4597 (cycle 9).

## H2H Evaluation Milestones
| Checkpoint | Samples | always_fold | calling_station | random_uniform | All beat? |
|------------|---------|-------------|-----------------|----------------|-----------|
| time_280 | 288 | +75.00 ✓ | -425.20 ✗ | +823.00 ✓ | 2/3 |
| time_314 | 320 | +75.00 ✓ | **+138.30** ✓ | +1026.70 ✓ | **3/3 YES** |

**MILESTONE**: At 320 samples, the model beats all 3 baselines. Calling station
flipped from -425 to +138 bb/100 in 32 samples. Variance still high (±202),
need more hands/training for statistical significance.

## NAS Backup
- Snapshot: `/root/work/backups/texas-holdem/snapshot-20260711T141826.tar.gz` (2.9MB)
- The project lives on the NAS-mounted `/root/work/` so all training
  checkpoints in `/root/work/software/texas-holdem/models/` are automatically
  on NAS.

## Notes
- Self-play is the bottleneck (~8-12 min per cycle of 32 samples). Training
  on NPU is fast; the per-state CPU inference during MCCFR traversal dominates.
- Local rclone to `nm-aihuanxin:` S3 hangs (XML decode error on ListBuckets).
  Use the chunked base64 upload via huanxin_shell, or rely on the NAS mount.
