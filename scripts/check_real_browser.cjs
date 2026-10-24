// Real-model browser acceptance; expected values come from direct inference, never fixtures.
const fs = require("node:fs/promises");
const path = require("node:path");
const { createRequire } = require("node:module");

async function main() {
  const runtimeRequire = createRequire(path.join(process.env.MEDINSPECT_NODE_MODULES, "package.json"));
  const { chromium } = runtimeRequire("playwright");
  const options = { headless: true };
  if (process.env.MEDINSPECT_BROWSER_PATH) options.executablePath = process.env.MEDINSPECT_BROWSER_PATH;
  else if (process.env.MEDINSPECT_BROWSER_CHANNEL !== "chromium")
    options.channel = process.env.MEDINSPECT_BROWSER_CHANNEL || "chrome";
  const cases = JSON.parse(await fs.readFile("artifacts/integration/cases.json", "utf8"));
  const browser = await chromium.launch(options);
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
  page.setDefaultTimeout(60000);
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  async function select(label, value) {
    await page.locator('[data-testid="stSelectbox"]').filter({ hasText: label })
      .getByRole("combobox").click();
    await page.getByRole("option", { name: value, exact: true }).click();
  }
  try {
    await page.goto(process.env.MEDINSPECT_BASE_URL, { waitUntil: "networkidle" });
    await page.getByRole("heading", { name: "Inspection workspace", exact: true }).waitFor();
    await page.getByLabel("Batch ID", { exact: true }).fill("REAL-INTEGRATION");
    for (let index = 0; index < cases.length; index++) {
      const sample = cases[index];
      await page.getByLabel("Instrument ID", { exact: true }).fill(sample.instrument_id);
      await select("Instrument class", sample.instrument_class);
      await page.locator('input[type="file"]').setInputFiles(sample.path);
      const analyze = page.getByRole("button", { name: /Analyze instrument$/ });
      await analyze.waitFor();
      await analyze.click();
      await page.getByRole("heading", { name: "Analysis result", exact: true }).waitFor({ timeout: 300000 });
      const metrics = page.locator('[data-testid="stMetricValue"]');
      const values = await metrics.allTextContents();
      if (!values.some(value => Math.abs(Number(value) - sample.anomaly_score) < 0.000002))
        throw new Error(`Displayed score differs for ${sample.source_path}: ${values}`);
      if (!values.some(value => Math.abs(Number(value) - sample.threshold) < 0.000002))
        throw new Error(`Displayed threshold differs for ${sample.source_path}: ${values}`);
      const status = sample.requires_review ? "AI screening: flagged for human review" :
        "AI screening: below the review threshold. An operator decision is still required.";
      await page.getByText(status, { exact: true }).waitFor();
      if (index === 0) {
        await select("Instrument class", "instrument19");
        await page.getByText("Image ready for inspection", { exact: true }).waitFor();
        await page.getByRole("heading", { name: "Analysis result", exact: true })
          .waitFor({ state: "hidden" });
        await select("Instrument class", sample.instrument_class);
        await analyze.click();
        await page.getByRole("heading", { name: "Analysis result", exact: true }).waitFor({ timeout: 300000 });
      }
      await select("Operator decision", sample.operator_decision);
      await page.getByLabel("Operator note", { exact: true }).fill(sample.operator_note);
      await page.getByRole("button", { name: /Save inspection$/ }).click();
      await page.getByText("Inspection saved in this session. Download it from Session report.", { exact: true }).waitFor();
      await page.screenshot({ path: `artifacts/integration/real-${index + 1}.png`, fullPage: true });
      console.log(`Browser analysis/review saved: ${sample.source_path}`);
      if (index < cases.length - 1) await page.getByRole("button", { name: /Start new inspection$/ }).click();
    }
    await page.getByRole("tab", { name: "Session report", exact: true }).click();
    const downloadEvent = page.waitForEvent("download");
    await page.getByRole("button", { name: /Download inspection CSV$/ }).click();
    await (await downloadEvent).saveAs("artifacts/integration/inspections.csv");
    await page.screenshot({ path: "artifacts/integration/report.png", fullPage: true });
    if (errors.length) throw new Error(errors.join("\n"));
  } catch (error) {
    await page.screenshot({ path: "artifacts/integration/failure.png", fullPage: true });
    await fs.writeFile("artifacts/integration/failure.txt", await page.locator("body").innerText());
    await fs.writeFile("artifacts/integration/failure.html", await page.content());
    throw error;
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
