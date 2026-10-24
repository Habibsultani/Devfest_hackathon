# Streamlit deployment

The app is prepared for Streamlit Community Cloud. Deployment is pending a real inference implementation, verified artifact, compatible backend dependency set, and a connected Streamlit account. No public demo has been verified.

## Release preparation

1. Integrate Person One's `src/inference.py` and transforms on a reviewed branch. Add their exact, working inference dependencies to `requirements.txt`; the current file pins only UI dependencies. Use the same Python version in testing and hosting.
2. Complete and verify `models/model_config.json`. Supply a stable HTTPS direct artifact URL and its actual SHA-256 digest. Keep credentials out of this file. Do not commit a checkpoint blob.
3. The local configuration is ignored to keep incomplete development metadata out of releases. After review, explicitly add the nonsecret release configuration with `git add -f models/model_config.json`. Alternatively, commit a completed release JSON under a new name and configure `MEDINSPECT_MODEL_CONFIG` to point to it. Do not deploy the example configuration.
4. Run `python -m scripts.fetch_model`, `python -m pip check`, `python -m pytest -q`, `python -m ruff check .`, and `python -m ruff format --check .`. Run genuine unseen-image tests separately. Reviewers should compare application outputs with direct model outputs.
5. Review changed files for datasets, artifacts, credentials, and inspection exports. Open a PR from `feature/web-app` into `main`; the teammate reviews before merge. Model changes can arrive through `feature/ai-model`.

Only trusted model artifacts should be loaded. The downloader verifies HTTPS and SHA-256 before installation, never accepts a model upload, rejects mismatched local files, and limits downloads to 1 GB. That limit is an application download guard, not evidence that a model fits Community Cloud's available memory.

## Community Cloud settings

From [Streamlit Community Cloud](https://share.streamlit.io/), connect the GitHub account and create an app:

| Setting | Value |
| --- | --- |
| Repository | `Habibsultani/Devfest_hackathon` |
| Branch | `main` after reviewed integration |
| Main file path | `app.py` |
| Python | Match the validated integrated environment, 3.11 or 3.12 |
| Dependencies | Root `requirements.txt`, including Person One's verified backend dependencies |
| Optional configuration override | Root secret `MEDINSPECT_MODEL_CONFIG = "models/release_config.json"` |

Streamlit exposes root-level secrets as environment variables. Keep real secrets in Community Cloud settings, never in Git. The current artifact downloader supports public or pre-authorized HTTPS URLs without custom authentication headers. Prefer a durable public artifact release whose licensing allows redistribution. An expiring signed URL will need replacement before it expires.

At the first analysis, the app checks the artifact and downloads it if missing and a URL is configured. A cached gateway shares the verified backend. Person One's inference implementation must cache the actual model. Review cloud logs for dependency, download, deserialization, CPU, and memory failures. If the service cannot run the trained model, reduce the artifact/resource requirements with Person One or use a verified alternative host. An interface-only deployment does not meet the submission gate.

## Public acceptance

Use a private browser window without developer authentication:

- Upload a genuinely unseen normal image. Confirm a real score/map and record any false alarm honestly.
- Upload a genuinely unseen defective image. Compare the real map with ground truth. Do not promise every defect will be detected.
- Confirm the supported instrument class manually. Reject unknown classes.
- Record a human decision with batch/instrument IDs, then download and inspect the CSV.
- Change the image and IDs. Confirm the previous result disappears until a new analysis completes.
- Restart the app. Confirm the saved model reloads without fitting and new inference works.
- Confirm a fresh session starts with no previous user's inspection records.
- Check upload failures and unavailable-model failures without fabricated predictions.

After all checks pass, add the **verified** public URL to the README and submission materials. Record the deployed Git commit, configuration digest, artifact digest, Python/backend package versions, and observed inference latency. Do not invent a deployment URL before publication.

Official references: [deployment settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [app dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), [secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management), [resource cache](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource).
