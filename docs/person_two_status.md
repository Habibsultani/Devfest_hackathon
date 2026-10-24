# Person Two delivery status

This branch implements application and deployment work from the technical development plan. It does not implement Person One's model, dataset, or evaluation responsibilities.

| Plan item | Status | Evidence or remaining dependency |
| --- | --- | --- |
| Git collaboration and shared app folders | Prepared | `feature/web-app`, Python ignore rules, PR template, CI checks. Existing repository reused. Collaborator invitations and account setup remain account actions. |
| Streamlit upload and results UI | Implemented | `app.py`, upload validation, explicit unavailable-model state |
| Heatmap visualization | Implemented | `src/visualization.py`, fixed model scale, overlay opacity, spatial checks |
| Human QC review and CSV | Implemented | `src/reporting.py`, independent decisions, UTC timestamps, session export |
| Error and unsupported-input handling | Implemented | Size/format/class/metadata checks and strict model-output validation |
| Integration with Person One | Boundary implemented | Real integration awaits P1's module, artifact, calibrated metadata, dependencies, and unseen examples |
| Deployment | Prepared, not published | Deployment instructions and artifact verification. Live release awaits genuine model integration and Streamlit account access. |
| Architecture and README | Implemented | `docs/architecture.md`, `docs/inference_contract.md`, `README.md` |
| Presentation | Application-stage deck and editable source | Model results and public demo evidence must come from the integrated project before submission. |
| Demo video | Recording script prepared | An honest actual-app inference recording requires the real model and verified public URL. |
| Submission | Checklist prepared | Team must fill factual model/evaluation evidence, record demo, verify form limits, and approve final materials. |

## Person One handoff

Supply `src/inference.py` exposing `inspect_image(PIL_RGB_image)`, exact preprocessing and cached loading, the saved memory bank/checkpoint, checksum and stable URL, model class/version, validation-selected threshold, fixed heatmap display bounds, compatible dependency versions, and held-out normal/defective test examples. See [the complete contract](inference_contract.md).

Do not create placeholder dataset or evaluation reports just to fill the tree. Those are Person One's measured deliverables. Application tests use isolated fixtures and do not count as real inference or model evaluation.

## Verification completed

On 24 October 2026, the application passed 62 tests on Python 3.12.14, dependency consistency checks, Ruff lint, and formatting checks. A running Streamlit instance returned a healthy HTTP response. Headless Chrome checks verified the missing-model state, disabled analysis, damaged-upload rejection, empty session report, and mobile rendering. All seven presentation slides were rendered and inspected, and the PPTX passed package, geometry, font, and reimport checks. Python 3.11 checks are configured in CI but were not run locally.

## Work remaining after handoff

Run the genuine model/UI comparison and unseen-image checks, verify restart behavior, merge reviewed features, deploy publicly, validate in a private browser session, update the deck with actual evidence, record the demo, and confirm all submission materials with the teammate. No public deployment, model result, demo video, or submitted form is claimed by this branch.
