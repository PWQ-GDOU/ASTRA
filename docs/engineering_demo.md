# Engineering Lifetime Demo

This demonstration turns the locked FEMTO to COMSOL v5 evidence into an
engineering-facing lifetime-status replay. It reads existing audit artifacts and
prediction traces; it never retrains a model or changes the locked result.

## Evidence boundary

The target is a COMSOL reaction-wheel degradation simulation proxy. The v5
experiment is labelled exploratory_post_hoc_replication and requires a new,
unseen outer-holdout confirmation before it can be presented as submission-ready
or operational flight evidence. The dashboard itself repeats this boundary.

The dashboard uses the calibrated transfer ensemble for the two strict positive
evidence configurations:

| Target fit groups | Transfer RMSE | Gain vs matched scratch | Outer-fold wins |
|---|---:|---:|---:|
| N=2 | 4.974 cycles | 7.30% | 4/5 |
| N=3 | 6.149 cycles | 6.51% | 3/5 |
| Predeclared aggregate | 6.233 cycles | 6.46% | 10/15 |

## Run locally

Build the static dashboard without starting a server:

    D:/tools/python_3_11_6/python.exe scripts/run_engineering_demo.py --no-serve

The generated offline artifact is outputs/engineering_demo/femto_ims_to_comsol_v5/index.html.
Open index.html directly in a browser, or run a local server for the same artifact:

    .\scripts\run_engineering_demo.ps1 -BindHost 127.0.0.1 -Port 8090

Then browse http://127.0.0.1:8090/.

## Run with Docker Desktop

The engineering-demo service uses the locked CPU image and mounts data/ read-only
plus outputs/ read-write. It serves the generated static dashboard at port 8090:

    docker compose up --build engineering-demo

For a build-only Docker run, which leaves the static files in outputs/:

    docker compose run --rm engineering-demo python scripts/run_engineering_demo.py --no-serve

## Operator replay and audit replay

Normal operator replay exposes only the predicted RUL trace. The slider steps
through telemetry endpoints, and the status panel computes a demonstration risk
state from the prediction. It does not display the held-out reference RUL.

Enable Audit replay only for validation review. That mode overlays the held-out
reference trace and reveals absolute error. The audit label is not an input to
the estimate, and it is deliberately absent from the normal operator status view.

The status thresholds are a demonstration policy only: critical at 30 cycles,
warning at 75 cycles, and watch at 150 cycles. A mission owner must approve
telemetry quality gates, thresholds, maintenance actions, and release criteria.

## Audit outputs

Each build emits the following files:

- index.html: self-contained, offline-capable operational replay
- DASHBOARD_DATA.json: immutable display data extracted from v5 artifacts
- ENGINEERING_DEMO_REPORT.json: build validation and SHA-256 hashes of all source audit files

The builder rejects missing artifacts, failed strict-evidence rows, missing
outer trajectories, finite-value violations, preflight leakage-guard failures,
and non-v5 exploratory protocol designations.
