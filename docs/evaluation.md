# Model evaluation and limitations

`reports/protocol.json` records the balanced protocol and manifest hashes written
before either selected class was fitted. Results are per instrument with an
equal-weight macro average in `reports/balanced_metrics.json`. Earlier results
remain under `reports/baseline/`.

The active model is the researched Dinomaly/SAM/tiling/LoRA integration. Its
frozen training protocol is `reports/dinomaly_protocol.json`, validation history
is `models/dinomaly/training_history.json`, and measured comparison is
`reports/dinomaly_metrics.json`. Validation selected step 200 from five fixed
checkpoints during 1,000 training steps. Both instruments reached validation F1
1.0 with zero false alarms. In the 16 balanced comparison images, the selected
model produced TP=7, FP=1, TN=7, FN=1, macro F1 0.875. Pixel AUROC was 0.9941
and 0.9785. This replacement was introduced after the earlier test inspection,
so these scores are development evidence, not a new untouched evaluation.

Each optimization batch contains equal instrument counts, sampling a normal
source image uniformly within class and then one of its native patches.
Frozen DINOv2-reg Small weights receive rank-8 qkv/output adapters. Anomalib's
Dinomaly decoder, bottleneck and cosine reconstruction loss are reused; LoRA
parameters and the reconstruction modules are optimized. BF16 training uses FP32
positive-kernel accumulation with epsilon for stability; inference uses FP32.
MobileSAM prompts and masks derive from image pixels only, without defect masks.
Validation-only F1 selects class thresholds, with ties preferring lower FPR.
CPU reload and original-size maps are verified. CUDA latency reports use warm
SAM-mask caches; cold-image SAM processing and CPU/cloud runtime must be
benchmarked separately. See `research_sources.md` for deviations and attribution.

The retained PatchCore baseline uses genuine Anomalib 2.2.0 PatchCore, an ImageNet-pretrained ResNet18,
layer2/layer3 features, a 5% k-center-greedy coreset and nine-neighbor weighted
image scoring. No neural-network fine-tuning is performed. Artifacts save the
extractor weights and feature bank as a tensor state dictionary, loaded with
`weights_only=True` after SHA-256 verification. The pretrained timm model is
`resnet18.a1_in1k`; the downloaded snapshot was
`491b427b45c94c7fb0e78b5474cc919aff584bbf`.

Each instrument has 12 normal training images, four normal plus four defect
validation images, and four normal plus four defect test images. All stages use
shared aspect-preserving 256-pixel letterboxing and ImageNet normalization.
Related source groups cannot cross folds.

Thresholds maximize validation balanced accuracy, `(recall + specificity) / 2`.
Candidates are distinct validation scores plus one value above their maximum.
Ties prefer lower false-positive rate, then smaller threshold. Validation
predictions, labels and threshold are saved in each model's `training_record.json`.
No final-test score selects a threshold, seed or class. An all-negative operating
point is reported honestly if that is what validation selects; zero false alarms
alone is not evidence of a useful screen.

The measured balanced baseline selected all-negative operating points and missed
4/4 defects for each instrument, with 0/4 false alarms for each. Image AUROC was
0.500 and 0.875, respectively. The optional region variant detected 2/4 and 3/4
defects with 0/4 and 3/4 false alarms. Neither result establishes a usable screen;
the baseline remains the explicit default and the variant remains experimental.

Reports include TP, FP, TN, FN, recall, precision, F1, false-positive rate and
image AUROC. Undefined metrics are JSON null. Raw anomaly distances are not
probabilities. Localization is pooled pixel AUROC at a maximum side of 256:
masks use grayscale >127 and nearest-neighbor resizing, maps use bilinear
resizing, normal masks are zero and padding is excluded. This ranking metric
can miss tiny defects and does not establish segmentation accuracy.

CPU latency covers preprocessing, forward inference and original-size map
alignment after a validation warmup; decoding and model loading are excluded.
Exact evaluated configurations are saved with reports, preserving provenance
when artifact retrieval URLs are later added. Confusion and ROC plots are
committed; source-image overlays remain local.

Robustness compares paired baseline, brightness 0.8 and 1.2, and JPEG quality 60,
with a fixed model and threshold. It does not isolate background changes. The
optional instrument-region experiment uses an independent RGB color heuristic
for blue/purple cloth, without ground-truth defect masks. It suppresses background
before the same normal-only PatchCore procedure. This experiment was introduced
after seeing the balanced baseline's test results, so its same-test comparison
is development evidence, not a new untouched final evaluation. Model selection
for that experiment must use validation performance only.

Four normal test images mean one false alarm changes per-instrument FPR by 25
percentage points. Defect variants share few source captures, masks and images
are simulated/controlled, and background/camera shifts can dominate features.
Balance does not remove these limitations. Preserve failed experiments and use
new independent data for future final validation. This prototype does not
certify device quality, safety or sterility.
