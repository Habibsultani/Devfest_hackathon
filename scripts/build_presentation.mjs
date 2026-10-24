// Build the seven-slide application-stage deck with the supplied Artifact Tool runtime.
// Optional tooling only. See docs/presentation.md for environment variables.
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const packages = process.env.MEDINSPECT_NODE_MODULES;
const skill = process.env.MEDINSPECT_PRESENTATIONS_SKILL;
const python = process.env.MEDINSPECT_RUNTIME_PYTHON;
if (!packages || !skill || !python) {
  throw new Error("Set the tooling environment variables described in docs/presentation.md.");
}
// The skill's import validator inherits this variable in its child process.
process.env.RUNTIME_NODE_MODULES = packages;
const { Presentation, PresentationFile } = await import(pathToFileURL(
  path.join(packages, "@oai/artifact-tool/dist/artifact_tool.mjs"),
).href);
const { finalizePresentation } = await import(pathToFileURL(
  path.join(skill, "container_tools/artifact_tool_utils.mjs"),
).href);

const workspace = process.cwd();
const output = path.resolve(process.env.MEDINSPECT_DECK_OUTPUT || "docs/MedInspect_AI_Application_v2.pptx");
const staging = await fs.mkdtemp(path.join(workspace, "artifacts/presentation-build-"));
await fs.mkdir(staging, { recursive: true });
await fs.mkdir(path.dirname(output), { recursive: true });
const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const palette = { ink: "#1B3039", teal: "#0F766E", muted: "#536873", paper: "#F6F8F9" };

function text(slide, value, x, y, w, h, size = 27, color = palette.ink, bold = false) {
  const shape = slide.shapes.add({
    geometry: "textbox", position: { left: x, top: y, width: w, height: h },
    fill: "none", line: { fill: "none", width: 0, style: "solid" },
  });
  shape.text = value;
  shape.text.style = {
    typeface: "Arial", fontSize: size, color, bold,
    verticalAlignment: "top", insets: 0,
  };
  return shape;
}

function slide(title, notes = "") {
  const item = deck.slides.add();
  item.background.fill = palette.paper;
  text(item, title, 70, 54, 1140, 75, 44, palette.ink, true);
  item.speakerNotes.text = notes + "\nSource: MedInspect_AI_Technical_Development_Plan.docx supplied by the team. " +
    "Application evidence: app.py, src/, tests/, docs/person_two_status.md in this repository. " +
    "Status: application stage, real model integration and public demo pending.";
  return item;
}

let s = deck.slides.add();
s.background.fill = palette.ink;
text(s, "MedInspect AI", 80, 176, 1120, 100, 76, "#FFFFFF", true);
text(s, "AI-assisted surgical instrument inspection", 84, 302, 1080, 60, 34, "#DCECF0");
text(s, "DevFest Samsun 2026", 84, 72, 1000, 50, 25, "#83D4D9");
text(s, "Person One: AI and dataset\nPerson Two: application and deployment", 84, 444, 1080, 92, 27, "#FFFFFF");
text(s, "Research prototype with human review", 84, 620, 1080, 40, 23, "#B6D1DD");
s.speakerNotes.text = "Source: team-supplied technical development plan. Team names were not provided. " +
  "This deck reflects application-stage work. The team must supply genuine model and evaluation evidence before submission.";

s = slide("Visible instrument inspection");
text(s, "A reviewable record\nfor each image", 72, 190, 460, 200, 44, palette.teal, true);
text(s, "Inspection context", 620, 170, 570, 48, 29, palette.ink, true);
text(s, "The operator identifies the batch, instrument, and supported class before screening.", 620, 232, 570, 125);
text(s, "Human assessment", 620, 408, 570, 48, 29, palette.ink, true);
text(s, "The operator reviews suspicious regions and records a decision with the model evidence.", 620, 470, 570, 130);

s = slide("Application and model boundary");
text(s, "Person Two", 72, 166, 510, 48, 30, palette.teal, true);
text(s, "Upload and metadata\n\nArtifact verification\n\nHeatmap and operator review\n\nSession CSV export", 72, 234, 510, 390, 28);
text(s, "Person One", 676, 166, 510, 48, 30, palette.teal, true);
text(s, "Saved PatchCore memory bank\n\nCached loading and transforms\n\nReal score and aligned map\n\nValidation threshold and evaluation", 676, 234, 510, 390, 28);
text(s, "Shared API: inspect_image(PIL RGB image)", 72, 650, 1120, 36, 24, palette.muted);

s = slide("Dataset and PatchCore", "Planned dataset: https://huggingface.co/datasets/purdueviperlab/Surgical_Instruments_Unsupervised_Defect_Detection\n" +
  "PatchCore reference: https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html\n" +
  "No dataset downloads, class selection, model training, or measured performance are claimed by Person Two.");
text(s, "Planned approach", 72, 172, 520, 50, 30, palette.teal, true);
text(s, "Purdue surgical instrument dataset\n\nOne supported class for the prototype\n\nPatchCore normal feature memory bank", 72, 246, 520, 300, 28);
text(s, "Person One's evidence", 684, 172, 520, 50, 30, palette.teal, true);
text(s, "Selected class and split counts\n\nUsage restrictions and simulated defects\n\nSaved artifact and calibrated threshold", 684, 246, 520, 300, 28);
text(s, "Dataset preparation and real model integration remain pending in this branch", 72, 620, 1130, 65, 25, palette.muted);

s = slide("Inspection workflow", "Screenshot: actual local app in its awaiting-model state, captured by scripts/check_browser.cjs. " +
  "The screenshot does not demonstrate a trained model or anomaly prediction.");
text(s, "1   Identify the instrument\n\n2   Upload and analyze\n\n3   Review the score and map\n\n4   Record a human decision\n\n5   Export the session CSV", 72, 190, 440, 390, 27);
const screenshot = path.join(workspace, "artifacts/browser/app-awaiting-model.png");
try {
  const bytes = await fs.readFile(screenshot);
  s.images.add({
    blob: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    contentType: "image/png", alt: "Actual application awaiting a trained model",
    fit: "contain", position: { left: 554, top: 154, width: 656, height: 478 },
  });
  text(s, "Actual app UI while model integration is pending", 554, 650, 656, 40, 23, palette.muted);
} catch (error) {
  if (error.code !== "ENOENT") throw error;
  text(s, "The app enables analysis after Person One supplies a verified trained model.", 590, 240, 560, 260, 36, palette.teal, true);
}

s = slide("Evaluation status");
text(s, "Model results pending", 72, 182, 1100, 80, 49, palette.teal, true);
text(s, "Application checks", 72, 316, 520, 48, 29, palette.ink, true);
text(s, "Workflow tests cover uploads, API validation, overlays, review, and CSV. Launch, readiness, and source packaging tools are ready.", 72, 384, 520, 200, 28);
text(s, "Required from held-out images", 684, 316, 520, 48, 29, palette.ink, true);
text(s, "TP / FP / TN / FN, recall, precision, F1, false-positive rate, AUROC, localization, and measured latency.", 684, 384, 520, 200, 28);
text(s, "UI fixtures establish application behavior. Real inference and accuracy await Person One's model.", 72, 626, 1130, 65, 24, palette.muted);

s = slide("Release gates and limits");
text(s, "Before a public demo", 72, 172, 520, 48, 30, palette.teal, true);
text(s, "Integrate the real model and dependencies\n\nVerify unseen-image inference and reload\n\nDeploy and check the complete review flow\n\nRecord the actual app and export", 72, 244, 520, 355, 27);
text(s, "Limits to communicate", 684, 172, 520, 48, 30, palette.teal, true);
text(s, "Lighting and background can affect results\n\nOnly the supported class can be screened\n\nHistory lasts for the current session\n\nNo safety or sterility certification", 684, 244, 520, 355, 27);
text(s, "The current branch has no verified public inference link", 72, 643, 1130, 40, 25, palette.muted);

const candidate = path.join(staging, "candidate.pptx");
await (await PresentationFile.exportPptx(deck)).save(candidate);
for (const [index, item] of deck.slides.items.entries()) {
  const png = await deck.export({ slide: item, format: "png", scale: 1 });
  await fs.writeFile(path.join(staging, `slide-${index + 1}.png`), new Uint8Array(await png.arrayBuffer()));
}
const result = await finalizePresentation({
  workspaceDir: workspace,
  candidatePath: candidate,
  finalPath: output,
  pythonExecutable: python,
  integrityValidatorPath: path.join(skill, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skill, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-heading-fit"],
  explicitTotalSlideCount: 7,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  fontPolicy: { basis: "design", families: ["Arial"] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(staging, "validation.json"),
});
console.log(JSON.stringify({ output, validation: result }));
