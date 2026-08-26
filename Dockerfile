# Deterministic CPU image for the audited FEMTO/IMS -> COMSOL reproduction.
FROM python:3.11.9-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    MPLBACKEND=Agg \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY docker/requirements.lock.txt /tmp/requirements.lock.txt

RUN pip install --no-cache-dir --upgrade pip==24.2 \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1+cpu \
    && pip install --no-cache-dir -r /tmp/requirements.lock.txt \
    && pip check

COPY . /workspace

CMD ["python", "scripts/reproduce_femto_ims_to_comsol.py", "--phase", "all", "--data-root", "/workspace/data", "--output-root", "/workspace/outputs/reproducibility/femto_ims_to_comsol"]
