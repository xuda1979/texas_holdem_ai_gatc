#!/usr/bin/env node

const { launchPersistentContext } = require('/Users/daxu/software/quantum-gpt/browser-automation/huanxin_browser_launch');
const {
  ensureProfileDir,
} = require('/Users/daxu/software/quantum-gpt/browser-automation/huanxin_profile');
const {
  bridgePageViaSafariSso,
} = require('/Users/daxu/software/quantum-gpt/browser-automation/huanxin_repair_profile_via_safari_sso');

const TRAIN_DEV_URL =
  'https://aihuanxin.cn/kunlun/kl-web?poolId=1&projectId=3ed7854b946a47b1a49ad754baa76cd3#/train-dev';

function parseArgs(argv) {
  const envName = argv[0];
  if (!envName) {
    throw new Error('Usage: node scripts/huanxin_ensure_env_started.js <envName>');
  }
  return { envName };
}

async function detectPageState(page) {
  const url = page.url();
  const bodyPreview = ((await page.locator('body').innerText().catch(() => '')) || '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 500);
  if (url.includes('/openid-connect/auth')) {
    return { state: 'login_required', url, bodyPreview };
  }
  return { state: 'app_surface', url, bodyPreview };
}

async function firstActionableButton(row, names) {
  for (const name of names) {
    const button = row.getByRole('button', { name }).first();
    const exists = (await button.count().catch(() => 0)) > 0;
    if (!exists) {
      continue;
    }
    const enabled = await button.isEnabled().catch(() => false);
    if (enabled) {
      return { name, button };
    }
  }
  return null;
}

async function main() {
  const { envName } = parseArgs(process.argv.slice(2));
  const { profileDir } = ensureProfileDir();
  const launch = await launchPersistentContext(profileDir);
  const context = launch.context;
  const page = context.pages()[0] || (await context.newPage());
  page.setDefaultTimeout(30000);

  try {
    await page.goto(TRAIN_DEV_URL, { waitUntil: 'networkidle', timeout: 180000 });
    await page.waitForTimeout(3000);

    let pageState = await detectPageState(page);
    if (pageState.state === 'login_required') {
      const repair = await bridgePageViaSafariSso(page, {
        authUrl: pageState.url,
        waitMs: 10000,
        resultPath: `/tmp/huanxin-${envName}-ensure-env-started-sso.json`,
      });
      if (!repair.ok) {
        throw new Error(
          `Safari SSO bridge failed for ${envName}. Final state=${repair.finalUrlState}.`
        );
      }
      await page.waitForTimeout(3000);
    }

    const row = page.locator('tr', { hasText: envName }).first();
    await row.waitFor({ state: 'visible', timeout: 60000 });

    let lastRowText = '';
    let actionTaken = 'none';
    for (let attempt = 0; attempt < 60; attempt += 1) {
      lastRowText = ((await row.textContent().catch(() => '')) || '').replace(/\s+/g, ' ').trim();
      const action = await firstActionableButton(row, ['打开', '运行', '启动']);
      if (!action) {
        await page.waitForTimeout(2000);
        continue;
      }

      if (action.name === '打开') {
        console.log(
          JSON.stringify(
            {
              ok: true,
              envName,
              actionTaken,
              readyAction: action.name,
              rowText: lastRowText,
              browserMode: launch.browserMode,
              launchFallbackUsed: launch.fallbackUsed,
            },
            null,
            2
          )
        );
        return;
      }

      actionTaken = action.name;
      await action.button.click({ timeout: 15000 });
      await page.waitForTimeout(5000);
    }

    throw new Error(`Timed out waiting for ${envName} to become openable. row=${lastRowText}`);
  } finally {
    await context.close().catch(() => {});
  }
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
