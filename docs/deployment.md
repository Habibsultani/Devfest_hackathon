# Integrated application deployment

Live app: **https://medinspect-ai-devfest.streamlit.app/**.
The public workflow passed five original-image checks across both classes,
human review, CSV parity, class-change invalidation and an empty separate browser
session. This is integration acceptance, not a new accuracy evaluation. The
known instrument19 corrosion miss remains. Hosted server-reboot acceptance has
not been recorded.
Use Python 3.12, Git and the combined CPU dependency lock. Run README setup,
`python -m scripts.doctor --require-model`, the unit suite and
`python -m scripts.check_real_app` before publishing a release.

## Model retrieval

`python -m scripts.setup_research_models` retrieves the pinned MobileSAM weights.
`python -m scripts.fetch_model --all --github-auth` retrieves the shared Dinomaly
checkpoint from the team draft release. Both are SHA-256 verified. Set
`GITHUB_TOKEN` with repository access in the hosting environment, or provide the
verified local files. Git credentials are also supported for local development.
Do not store credentials in configuration JSON, source or a Docker image.

The source repository is public, but draft release assets still require access.
Use a dedicated fine-grained token restricted to `Devfest_hackathon`, owned by a
user with push access. Read-only access returned HTTP 403 for this release;
**Contents: Read and write** resolved the hosted download. Published release
downloads normally need only read access. Save changed permissions in GitHub,
then reboot the app. If the token value changes, update the Streamlit secret.

The app's error identifies the affected file and HTTP status. GitHub 401 indicates
a rejected credential; 403/404 can indicate repository selection or draft access.
Exhausted rate limits and separate MobileSAM host failures are reported distinctly.
Do not bypass checksum verification when troubleshooting a download.

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

Use these Streamlit Community Cloud settings:

| Setting | Value |
| --- | --- |
| Repository | `Habibsultani/Devfest_hackathon` |
| Branch | `feature/streamlit-deployment`; use `main` after reviewed changes merge |
| Main file | `cloud/streamlit/app.py` |
| Python | `3.12` |

The cloud entrypoint delegates to the root app. Its adjacent `requirements.txt`
selects the tested `requirements-lock.txt` and the PyTorch CPU index. Root
`packages.txt` supplies Git and OpenCV's Linux libraries; keep the root
`.streamlit/config.toml`.

In the hosting dashboard's **Settings → Secrets**, add a root-level value:

```toml
GITHUB_TOKEN = "<repository-scoped token, entered only in the dashboard>"
```

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
with measured results. Presentation and video are separate submission materials;
obsolete application-stage decks are no longer included in the source repository.

Official references: [dependency discovery](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies),
[deployment settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy),
and [secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).
