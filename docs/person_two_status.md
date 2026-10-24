# Integrated delivery status

The app and AI branches are combined. The interface uses the active Dinomaly +
MobileSAM model for instrument12 and instrument19, with class-specific thresholds.

| Deliverable | Status |
| --- | --- |
| Upload, class selection and real inference | Integrated; class/config passed explicitly |
| Original-size heatmap and overlay | Fixed validation scales saved in both configs |
| Human review and CSV export | Integrated; AI flag stays separate from human decision |
| Private model retrieval | GitHub authentication and SHA-256 for main and SAM weights |
| Combined environment | Python 3.12 CPU lock, compatible Streamlit/pandas/packaging |
| Automated checks | Unit, artifact, reload, app and browser checks; real-model browser/CSV script |
| Source delivery | Packager excludes weights, datasets, secrets and all virtual environments |
| Container | Combined dependencies and non-root startup; verified separately by CI |
| Public deployment | Pending hosting and public acceptance |
| Final deck, recording, submission | Pending team work; application-stage decks are historical |

See `reports/integration_smoke.json` for the latest five-image real-model application
acceptance and the README for reproduction commands. Browser screenshots/CSV are
local ignored artifacts. Integration acceptance does not establish new model accuracy.
The balanced development comparison remains 14/16 correct; the known instrument19
corrosion miss is retained in application testing.
