# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## What Goes Here

Things like:

- Camera names and locations
- SSH hosts and aliases
- Preferred voices for TTS
- Speaker/room names
- Device nicknames
- Anything environment-specific

## Examples

```markdown
### Cameras

- living-room → Main area, 180° wide angle
- front-door → Entrance, motion-triggered

### SSH

- home-server → 192.168.1.100, user: admin

### TTS

- Preferred voice: "Nova" (warm, slightly British)
- Default speaker: Kitchen HomePod
```

## Why Separate?

Skills are shared. Your setup is yours. Keeping them apart means you can update skills without losing your notes, and share skills without leaking your infrastructure.

---

Add whatever helps you do your job. This is your cheat sheet.

## Huanxin Training Environment

- Requested active environment name: **ai3** as of 2026-04-30.
- Previous active environment name: **AI**. Older notes and wrappers may still mention AI; verify live support before remote training.
- ai3 train-dev URL: https://aihuanxin.cn/kunlun/kl-web?poolId=6&projectId=21b4208dde424e96b159362ef49c9c96#/train-dev/environment/dl-c72bd81a96e33134bbe0ae4a478fbab0?name=ai3
- Previous AI train-dev URL: https://aihuanxin.cn/kunlun/kl-web?poolId=6&projectId=21b4208dde424e96b159362ef49c9c96#/train-dev/environment/dl-9a5a098accce31c28cf4c6ca23391341?name=AI
- Remote project location: `~/software/texas-holdem` inside the active Huanxin environment.
- Use `scripts/ai3_shell.sh`, `scripts/ai3_sync_from_s3.sh`, `scripts/ai3_job.sh`, and `scripts/ai3_push_results_to_s3.sh` for ai3 remote work after a live smoke command passes.
