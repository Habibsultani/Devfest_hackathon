# Application presentation

`MedInspect_AI_Application.pptx` follows the technical plan's seven-slide narrative. It accurately presents the current application stage, including the missing model and evaluation evidence. The workflow screenshot shows the actual app while it awaits a trained model. It contains no simulated heatmap or invented metric.

The deck is editable in PowerPoint. Before submission, update the model, dataset, and evaluation sections from Person One's actual evidence. Add the real team names and verified public demo. `scripts/build_presentation.mjs` is the editable authoring source.

## Optional rebuild tooling

The presentation and browser-check scripts are development tools separate from the Python app. They use a host-supplied Node runtime with `@oai/artifact-tool` and `playwright`. They do not add a Node dependency to Streamlit hosting. Set:

- `MEDINSPECT_NODE_MODULES`: absolute host tooling `node_modules` path.
- `MEDINSPECT_PRESENTATIONS_SKILL`: installed presentation skill directory containing `container_tools`.
- `MEDINSPECT_RUNTIME_PYTHON`: the host runtime Python executable used by deck validators.
- `MEDINSPECT_DECK_OUTPUT`: optional new output filename, relative to the project root or absolute within the workspace. The finalizer will not overwrite an existing deck.

With the app running locally, `node scripts/check_browser.cjs` captures the awaiting-model UI and validates basic browser behavior. Set `MEDINSPECT_BROWSER_PATH` if the browser is not installed as Chrome. Then run `node scripts/build_presentation.mjs`. It exports seven slide previews and validation records under ignored `artifacts/presentation/` and the final PPTX under `docs/` by default. Open and inspect every preview after any content/layout change.

The source deck contains citations in speaker notes. Model-performance evidence must be supplied by Person One rather than inferred from application tests.
