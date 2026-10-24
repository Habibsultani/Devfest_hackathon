// Local visual QA only. Uses the browser tools already supplied by the host.
// Usage: MEDINSPECT_NODE_MODULES=/path/to/node_modules node scripts/check_browser.cjs
const fs = require("node:fs/promises");
const path = require("node:path");
const { createRequire } = require("node:module");

async function main() {
  const modules = process.env.MEDINSPECT_NODE_MODULES;
  if (!modules) throw new Error("Set MEDINSPECT_NODE_MODULES to the browser tooling directory.");
  const runtimeRequire = createRequire(path.join(modules, "package.json"));
  const { chromium } = runtimeRequire("playwright");
  const launchOptions = { headless: true };
  if (process.env.MEDINSPECT_BROWSER_PATH) {
    launchOptions.executablePath = process.env.MEDINSPECT_BROWSER_PATH;
  } else {
    launchOptions.channel = "chrome";
  }
  const browser = await chromium.launch(launchOptions);
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("http://127.0.0.1:8501", { waitUntil: "networkidle" });
    await page.getByRole("heading", { name: "MedInspect AI", level: 1, exact: true }).waitFor();
    await page.getByText("Awaiting trained model", { exact: true }).waitFor();
    const analyze = page.getByRole("button", { name: "Analyze instrument", exact: true });
    if (!(await analyze.isDisabled())) throw new Error("Analysis must be disabled without a model.");
    await fs.mkdir("artifacts/browser", { recursive: true });
    await page.screenshot({ path: "artifacts/browser/app-awaiting-model.png", fullPage: true });
    await page.getByLabel("Batch ID", { exact: true }).fill("BATCH-001");
    await page.getByLabel("Instrument ID", { exact: true }).fill("INST-014");
    await page.locator('input[type="file"]').setInputFiles({
      name: "damaged.png", mimeType: "image/png", buffer: Buffer.from("not an image"),
    });
    await page.getByText("The image is unreadable, damaged, or too large to decode.").waitFor();
    if (!(await analyze.isDisabled())) throw new Error("Damaged images must never reach inference.");
    await page.screenshot({ path: "artifacts/browser/app-invalid-upload.png", fullPage: true });
    await page.getByRole("tab", { name: "Session report", exact: true }).click();
    await page.getByText("No inspections saved yet.", { exact: false }).waitFor();
    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole("tab", { name: "Inspect instrument", exact: true }).click();
    await page.waitForFunction(() => [...document.querySelectorAll('[role="tab"]')].some(
      (tab) => tab.textContent === "Inspect instrument" && tab.getAttribute("aria-selected") === "true"
    ));
    await page.waitForTimeout(500);
    await page.screenshot({ path: "artifacts/browser/app-mobile.png", fullPage: true });
    if (errors.length) throw new Error(errors.join("\n"));
    console.log("Browser checks passed: missing model, invalid upload, empty report, mobile render.");
  } finally {
    await browser.close();
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
