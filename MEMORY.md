# Long-Term Memory

- For the Texas Hold'em workspace, use the Huanxin environment named **AI** for remote execution and artifact storage.
- AI train-dev URL: https://aihuanxin.cn/kunlun/kl-web?poolId=6&projectId=21b4208dde424e96b159362ef49c9c96#/train-dev/environment/dl-9a5a098accce31c28cf4c6ca23391341?name=AI
- Keep the project under `~/software` inside the AI environment.
- Keep checkpoints and large model artifacts off the local machine; use S3 as the transfer path when moving code/results.
- Prefer the `quantum-gpt` Huanxin browser automation stack for Huanxin connectivity in this workspace.
- The existing Safari Huanxin train-dev tab is maintained by the keepalive LaunchAgent `com.quantumgpt.huanxin-safari-keepalive`.
- 2026-04-30: User emphasized fast, test-driven R&D: iterate quickly, run focused tests, fix concrete bugs, and seek innovative algorithm improvements that can be validated.
- 2026-04-30: User asked to use Huanxin `ai3`. ai3 URL observed from local probe artifacts: https://aihuanxin.cn/kunlun/kl-web?poolId=6&projectId=21b4208dde424e96b159362ef49c9c96#/train-dev/environment/dl-c72bd81a96e33134bbe0ae4a478fbab0?name=ai3. Older workspace notes still target `AI`, so verify live `ai3` wrapper support and keep checkpoints/model artifacts remote or in S3.
