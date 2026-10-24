# Integrated application deployment

The real model and app are combined. No public demo URL has been verified yet.
Use Python 3.12, Git and the combined CPU dependency lock. Run README setup,
`python -m scripts.doctor --require-model`, the unit suite and
`python -m scripts.check_real_app` before publishing a release.

## Model retrieval

`python -m scripts.setup_research_models` retrieves the pinned MobileSAM weights.
`python -m scripts.fetch_model --all --github-auth` retrieves the shared Dinomaly
checkpoint from the private team draft release. Both are SHA-256 verified. Set
`GITHUB_TOKEN` with repository access in the hosting environment, or provide the
verified local files. Git credentials are also supported for local development.
Do not store credentials in configuration JSON, source or a Docker image.

The app reads the committed `models/model_registry.json` by default. It verifies
both artifacts and can download missing weights on first analysis. Optional
`MEDINSPECT_MODEL_CONFIG` restricts a deployment to one reviewed config. The
example legacy configuration is intentionally incomplete and must not be deployed.
Artifact and segmentation-cache directories need writable storage. Corrupt local
weights are rejected; replace them with the verified matching files.

## Docker

```bash
docker build -t medinspect-ai:local .
docker run --rm -p 127.0.0.1:8503:8501 --env GITHUB_TOKEN medinspect-ai:local
```

Set the token in the host environment first. The image installs the combined CPU
lock and Git/OpenCV runtime dependencies. It runs as an unprivileged user. Model
blobs, datasets, virtual environments and secrets are excluded from the build.
The container downloads weights on first analysis unless verified files are
mounted at their configured locations. Keep the cache/artifact paths writable.
The HTTP health check confirms the web process; real inference must be checked
separately. GitHub Actions builds and checks the container. A local Docker daemon
is required for local builds.

## Hosted service

For Streamlit Community Cloud select the repository, reviewed `main` revision,
`app.py`, and Python 3.12. Install the combined CPU stack; `requirements.txt` pins
both direct model and UI dependencies, while the lock provides the full tested
set. A host using only direct dependencies must be checked against the lock.
Configure the private artifact token through the service's secret/environment
settings. Git must be available for the pinned MobileSAM dependency.

Native-resolution tiling and MobileSAM need substantially more resources than an
interface-only app. Measure cold-start, memory and inference time on the actual
host before choosing it. Use an adequate container host if shared hosting cannot
run this model. No hosted performance is claimed by the local acceptance report.

## Public acceptance

Use a fresh private browser session to upload new normal and defective examples,
check both manually selected instrument classes, inspect maps, record operator
decisions and verify exported CSV values. Record false alarms and misses honestly.
Change class/image/IDs and confirm stale results disappear. Restart the process
and verify offline checkpoint reload without fitting. Check that new sessions
cannot see previous session records. Download records before leaving: history is
session-only.

After public acceptance, record the verified URL, deployed commit, config/artifact
hashes, package versions and measured latency. Update the final deck and recording
with measured results. The existing application-stage slides are not final
integrated submission evidence.
