# Streamlit Community Cloud release

This entrypoint runs the existing root application with the tested CPU lock.
The root README and direct requirements stay unchanged. Community Cloud reads
the requirements beside its selected entrypoint before root requirements and
installs the root `packages.txt` for Git and OpenCV's Linux libraries.

## Deployment settings

| Setting | Value |
| --- | --- |
| Repository | `Habibsultani/Devfest_hackathon` |
| Branch | `feature/streamlit-deployment` for the first checked deployment |
| Main file | `cloud/streamlit/app.py` |
| Python | `3.12` |
| Suggested subdomain | `medinspect-ai` if available; the dashboard confirms the actual URL |

In **Advanced settings → Secrets**, set a root-level secret:

```toml
GITHUB_TOKEN = "<repository-scoped token, entered only in the hosting dashboard>"
```

The current checkpoint is an asset of a private draft GitHub release. Use a
dedicated token restricted to this repository with the permissions GitHub needs
to read its draft assets. Draft-release access may require write access; a normal
published private release can use read access. Do not paste the token into chat,
commit it, or reuse a broad personal credential. Keep both model configuration
files and thresholds from the integrated main revision.

The app retrieves and checksum-verifies its Dinomaly and MobileSAM artifacts on
first analysis. Allow the cold start to complete. A successful UI launch alone
does not establish a working model. If the service reports memory/resource
failures during real analysis, retain those logs and use a host with enough RAM.

## Acceptance before sharing

Open the actual deployed URL in a fresh private browser session. Select each
supported class, upload real normal and defective examples, inspect the aligned
map, save a human decision and verify the downloaded CSV. Check class/input
changes clear the old result. Verify a new session has no previous records and
reboot/reload works with the verified checkpoint. Record actual latency, observed
misses and false alarms. The known instrument19 corrosion miss remains a model
limitation. Use research-prototype wording.

After the deployed commit passes acceptance and teammate review, merge this
branch into `main` and change the hosting branch to `main`. Recheck the real
workflow after the host redeploys. No public URL is claimed by these files.

Official references: [dependency discovery](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies),
[deployment settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy),
and [secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).
