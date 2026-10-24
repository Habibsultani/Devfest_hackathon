# Dataset and balanced cohorts

The revised experiment uses Purdue instrument12 and instrument19, selected using
source counts and defect coverage before model inference on those classes. The
initial instrument1/2/3 run is summarized in `reports/baseline_summary.json`;
its full historical outputs remain available in that file's recorded Git revision.

Source: [Purdue VIPER dataset](https://huggingface.co/datasets/purdueviperlab/Surgical_Instruments_Unsupervised_Defect_Detection)
at immutable revision `e791b82e9616889e356b4cf7d40d6b56ba87b0c5`.
The downloader verifies each file size and source LFS SHA-256 and retains the
source inventory and dataset card locally.

| Instrument | Source normals | Source defects | Masks |
| --- | ---: | ---: | ---: |
| instrument12 | 27 | 14 | 14 |
| instrument19 | 24 | 17 | 17 |

The source contains blood, corrosion, crack, cut and pores for both classes.
The author's card declares CC-BY-NC-ND-4.0 and research use. Blood-like defects
are physically simulated; other defect types use TF-IDG. Dataset images, masks
and derived overlays stay local and are excluded from Git. This repository
asserts no commercial-use or modified-source redistribution permission.
Attribute [Huang et al, Unsupervised Defect Detection for Surgical Instruments](https://doi.org/10.1109/ICIP61757.2026.11630477).

| Split per instrument | Normal | Defect | Total |
| --- | ---: | ---: | ---: |
| Training | 12 | 0 | 12 |
| Validation | 4 | 4 | 8 |
| Test | 4 | 4 | 8 |

Both instruments have equal counts. Across them there are 24 normal training
images, 16 balanced validation images and 16 balanced test images. The active
Dinomaly model trains on normal patches; labeled validation selects thresholds.
Sampling is without replacement, with seed 42. Unused files are listed in each
`selection_audit.json`; no augmentation or duplication fills balance quotas.

Filename capture IDs, identical decoded-pixel hashes and 64-bit pHash distances
of at most 4 form connected source groups. Groups are assigned before sampling;
unused relatives are excluded rather than moved into another fold. Only original
training-folder normals can enter training. The source evaluation folder is
partitioned between validation and test before fitting. Ground-truth defect masks
never enter preprocessing or inference.

Image-count balance is not independent-capture balance. Instrument12 has three
linked abnormal groups and instrument19 has two. Some folds therefore include
several variants of one source capture. Group counts, dimensions, hashes and
exclusions accompany the reports. pHash is a leakage heuristic and cannot prove
independence of physical instruments. Subtype counts can differ because all
variants of a source photograph must stay together.

The full index checks readability, dimensions, mask pairing and group separation.
Masks with soft grayscale edges use foreground values above 127 in evaluation.
Manifests contain relative image/mask paths, class, label, defect type, split,
source group, dimensions and image/mask checksums. Report manifest copies are
exact provenance snapshots; executable manifests live in `data/processed/`.
