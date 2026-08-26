# Data Sources And Reproducibility Assets

This repository tracks the compact inputs required by the strict experiments.
Large binary inputs are Git LFS objects.  SHA-256 values make the data
selection auditable and allow the experiment manifests to reject mismatched
inputs.

## Versioned Data

| Repository path | Origin | SHA-256 |
|---|---|---|
| `data/raw/nozzle_ablation_full.csv` | Competition nozzle quasi-steady long-duration data | `2d1b4d906b8832e84f802eea449eb1ec0d744911604bd37cbfa70241e807a7c2` |
| `data/processed/nozzle_mt_200/nozzle_sim_200traj.csv` | Strict 200-trajectory nozzle simulation | `541c2fc894f089d4799bddab2cc48ce2a3d2f5a4b0573127d1a239838fc50c60` |
| `data/processed/nozzle_mt_200/nozzle_sim_200traj_metadata.json` | Strict 200-trajectory nozzle metadata | `db624dfa4b2f166e1f2e776a4924d724a4a10e453c7771e3045597616b6c227d` |
| `data/processed/nasa_battery/5. Battery Data Set/B0005.mat` | NASA Battery Aging data | `0eae4585baf3f200c09fe24c5ab884f1889679fc75206ca1aa19da704104f0b0` |
| `data/processed/nasa_battery/5. Battery Data Set/B0006.mat` | NASA Battery Aging data | `fa818ab4db5db8ab21e910b6dd6c3e20d3761bb9672089e1a4de8f96074616c5` |
| `data/processed/nasa_battery/5. Battery Data Set/B0007.mat` | NASA Battery Aging data; right-censored diagnostic | `d022afa086efaf54ab5b63f05220f5be178c8027e2fa8d68589db0e85a441a3b` |
| `data/processed/nasa_battery/5. Battery Data Set/B0018.mat` | NASA Battery Aging data | `d1e6c923a43ea1c9666b3a90bbb521757a067fd60b4d17dbfaa49c50b179da69` |
| `data/processed/ncmapss/N-CMAPSS_full.zip` | N-CMAPSS processed benchmark input | `121e547c56739cd7fe46468a2ec420297fd96c61a2813eb7af8e72f295060dab` |
| `data/raw/competition/ncmapss_ds01.zip` | Competition-provided N-CMAPSS DS01 archive | `e595250537f04bb4f374946dc85070d4635aba3d513ed32b3796a6448d239969` |
| `data/raw/competition/reaction_wheel_comsol_degradation.zip` | Competition-provided reaction-wheel COMSOL degradation data | `b628deb89634cf8732cd96bb62d675b820aedb0aa337fd36082db53bfebca8fe` |
| `data/processed/femto_bearing.zip` | FEMTO bearing data used only as the reaction-wheel mechanical-degradation proxy | `e21bb22bd8d54fd18ebe98b4b4e094c0c40469bda19811a2a642d5cc84ebd81f` |
| `data/processed/ims_processed/` | Processed public IMS source artifact supplied in the Weibull reference archive; `2nd_test` train and `3rd_test` validation | `887eea9cdb541ff40de8164f1913f9a37d22b12e3c48cc0eaa2d6dc7c100f4e6` |

The v3 experiment rechecks the nozzle, battery, and FEMTO hashes against
`outputs/cross_component_transfer_v3/DATA_MANIFEST.json`.

## Reference Material

| Repository path | Source |
|---|---|
| `docs/competition/competition_rules.pdf` | Competition rules supplied with this worktree |
| `third_party/weibull-knowledge-informed-ml-master.zip` | Supplied Weibull knowledge-informed ML reference archive |

## Deliberately External Source Archives

The DCU contains the broader NASA source archive
`data/raw/NASA_5_Battery_Data_Set.zip`
(`82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e`)
and several overlapping NASA subarchives.  They are not duplicated here:
the exact four MAT files used by the strict protocol are versioned above.

Run `git lfs pull` after cloning to materialize the large LFS datasets.

## Docker Reproduction

`Dockerfile` pins the Python 3.11 CPU runtime and
`docker/requirements.lock.txt` pins the experiment dependencies. The image
contains code only: `data/` is mounted read-only and `outputs/` is mounted
writable by `docker-compose.yml`. The reproduction driver recomputes the three
input hashes before each run and records package versions plus a code digest in
`REPRODUCTION_ENVIRONMENT.json`.
