# Docker Reproduction Report

## Run

- Date: 2026-08-21
- Protocol: FEMTO/IMS -> COMSOL strict transfer
- Phase: `all` (`preflight` -> `smoke` -> `full`)
- Canonical run ID: `femto-ims-to-comsol-20260821T010000Z`
- Canonical output: `outputs/reproducibility/femto-ims-to-comsol-20260821T010000Z/`
- Canonical full report:
  `outputs/reproducibility/femto-ims-to-comsol-20260821T010000Z/full/REPORT.json`
- Independent deterministic verification:
  `outputs/reproducibility/femto-ims-to-comsol-20260821T020000Z/full/`

## Environment

- Docker image: `astra-rul-femto-ims-comsol:py311-cpu`
- Image digest: `sha256:e44296ef716a86c05b2a99a4104d0bee6d1bddff3049b1408a4affc109ad1fc8`
- Base image digest: `sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317`
- Python: `3.11.9`
- PyTorch: `2.3.1+cpu`
- NumPy: `1.26.4`
- SciPy: `1.13.1`
- scikit-learn: `1.5.1`
- pandas: `2.2.2`
- h5py: `3.11.0`
- Platform: Docker Desktop Linux / WSL2
- Data mount: read-only
- Thread controls: `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
  `OPENBLAS_NUM_THREADS=4`, `NUMEXPR_NUM_THREADS=4`
- Determinism control: `PYTHONHASHSEED=0`
- Code SHA-256: `0ab544e9692574e265eccd6cca41b963993cdbe279b58fa6b3090565553c4c72`

The exact machine-readable environment record is
`outputs/reproducibility/femto-ims-to-comsol-20260821T010000Z/REPRODUCTION_ENVIRONMENT.json`.

## Input Audit

All three required inputs passed SHA-256 verification:

| Input | SHA-256 |
|---|---|
| `data/processed/femto_bearing.zip` | `e21bb22bd8d54fd18ebe98b4b4e094c0c40469bda19811a2a642d5cc84ebd81f` |
| `data/processed/ims_processed/` | `887eea9cdb541ff40de8164f1913f9a37d22b12e3c48cc0eaa2d6dc7c100f4e6` |
| `data/raw/competition/reaction_wheel_comsol_degradation.zip` | `b628deb89634cf8732cd96bb62d675b820aedb0aa337fd36082db53bfebca8fe` |

## Execution Result

- Preflight: passed
- Smoke: passed; 66 macro rows
- Full: passed; 120 macro rows
- Full supervisor attempts: 1
- Full report runtime: 190.06 seconds
- Full supervisor runtime: 210.25 seconds
- Source arms: `femto`, `ims`
- Full prediction rows: 6,744
- `pip check`: passed
- Container package import check: passed
- Full Docker test suite with the read-only data mount: `91 passed`

## Deterministic Verification

The source and target models now set their explicit seed before model
construction, reset the same seed before training, request deterministic
PyTorch algorithms, and disable CPU MKLDNN. A second independent full run used
the identical image and inputs.

| Artifact | Canonical SHA-256 | Independent-run SHA-256 | Equal |
|---|---|---|---|
| `MACRO.csv` | `75341b63e9cbd465a734a839adbf1cdb7f98daaf2627c31e34d3cb9c3ee5a73a` | `75341b63e9cbd465a734a839adbf1cdb7f98daaf2627c31e34d3cb9c3ee5a73a` | yes |
| `PREDICTIONS.csv` | `ce04fc15040daf68cf58fe456ce144b6294ea27677172d892ca72df01bef7c5a` | `ce04fc15040daf68cf58fe456ce144b6294ea27677172d892ca72df01bef7c5a` | yes |
| `PROTOCOL.json` | `906665720bd0eb9b4201ec2a1d9d9182f4087c781f6e10d0f05055359e967036` | `906665720bd0eb9b4201ec2a1d9d9182f4087c781f6e10d0f05055359e967036` | yes |
| `DATA_MANIFEST.json` | `05f731ce1aef86c8609cf44eee15f5d601f4428dd395a7032ff69f5b15156200` | `05f731ce1aef86c8609cf44eee15f5d601f4428dd395a7032ff69f5b15156200` | yes |

The full output contains `PREFLIGHT.json`, `PROTOCOL.json`,
`DATA_MANIFEST.json`, `REPORT.json`, `MACRO.csv`, `PREDICTIONS.csv`,
`SUPERVISOR_STATUS.json`, `experiment.log`, and
`REPRODUCTION_ENVIRONMENT.json`.

## Correction Record

The first container run (`femto-ims-to-comsol-20260821T000000Z`) successfully
completed the requested stages but exposed an existing protocol defect:
models were initialized before their nominal seed was set. It is retained as a
debug artifact only and is not used for numerical claims. The seeded-model
repair, its regression tests, and the two verified runs above replace it as the
canonical Docker evidence.

## Bootstrap Note

Docker Desktop could not reach Docker Hub directly because no HTTPS proxy was
configured. The identical Python base image was pulled through the configured
Docker mirror and locally tagged with the canonical
`python:3.11.9-slim-bookworm` name before building the unchanged Dockerfile.
The resulting image uses the recorded base-image digest above.
