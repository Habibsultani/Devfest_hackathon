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

The repository is public, but the checkpoint remains an asset of a draft GitHub
release. Public source access does not expose draft assets. Use a dedicated
fine-grained token owned by a repository collaborator with push access, restricted
to `Devfest_hackathon`. For the current draft release, grant **Contents: Read and
write** if read-only access returns HTTP 403/404. Published release downloads
normally require only read access. Keep the token in hosting Secrets, outside chat
and Git. Keep both configurations and thresholds from the integrated main revision.

If analysis reports a download failure, check its filename and HTTP status. A
GitHub 401 indicates an invalid/expired credential. GitHub 403/404 can indicate
repository selection or draft-release access; an exhausted rate limit is reported
separately. A MobileSAM download-host error is separate from GitHub API token
access. After changing token permissions, save them in GitHub and reboot the app;
if the token value changes, also replace the Streamlit secret. Never disable
checksum verification to work around a failed download.

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
