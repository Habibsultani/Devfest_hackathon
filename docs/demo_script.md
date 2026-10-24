# Demo recording and submission preparation

Record the actual integrated application after the public acceptance checks pass. The repository now includes real Dinomaly inference and measured development results. The five-image browser/CSV acceptance uses real weights. No public URL or final recording has been verified.

## Recording sequence

1. Open the verified public URL in a private browser window. Introduce MedInspect AI as a research prototype for visible-defect screening with human review.
2. Enter a real demonstration batch ID and instrument ID. Confirm the model's supported class. Upload an image held out from training and threshold selection.
3. Run analysis. Show the image-level score, validation threshold, real suspicious-region overlay, and model version. Explain that the score is not a defect probability.
4. Show an unseen normal example and an unseen defective example. Describe the actual outputs, including any false alarm or missed defect. Do not select only favorable examples and imply guaranteed performance.
5. Choose a human decision independently of the AI flag and add a useful note. Show an Inconclusive decision when the image does not support a confident review.
6. Save the inspection, open Session report, and download the CSV. Open the file and confirm the IDs, model version, score, AI flag, timestamp, and human decision match the demonstration.
7. Explain the limitations: lighting/background sensitivity, supported class, simulated-defect caveat documented in the dataset notes, no safety/sterility certification, and session-only history.

Do not include credentials, patient information, private data, or model-download secrets in the recording. Confirm the submission form's current duration and upload-size limits before choosing capture/compression settings. No numeric form limits are assumed here.

## Seven-slide narrative

The application-stage presentation follows the plan's seven topics: project/team roles, inspection problem, architecture/human review, Person One's planned dataset/model, app workflow, evaluation status, and limitations/next integration steps. Before submission, replace pending model/evaluation sections using Person One's actual evidence and add the verified public demo URL. Preserve the distinction between UI tests and model accuracy.

## Submission checklist

- [ ] Saved Dinomaly and MobileSAM artifacts reload in a fresh process without training.
- [ ] Dataset provenance, usage restrictions, splits, and simulated-defect caveat are documented by Person One.
- [ ] Real held-out evaluation and test counts exist in Person One's reports.
- [ ] New public-session uploads run genuine dynamic inference.
- [ ] Human review and the downloaded CSV work end to end.
- [ ] README includes working installation, artifact retrieval, exact backend dependencies, and verified live link.
- [ ] Slides contain actual model/evaluation evidence, source declarations, and current limitations.
- [ ] Actual-app recording is complete and respects current form limits.
- [ ] Source ZIP excludes `.git`, `.venv`, secrets, raw datasets, and checkpoint blobs and respects the form's size limit.
- [ ] Both team members review the final factual claims and approve the submission.

The team should submit through the actual event form after checking the current rules. This repository does not invent form URLs or claim a submission.
