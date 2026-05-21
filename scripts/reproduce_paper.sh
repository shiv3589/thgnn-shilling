#!/usr/bin/env bash
# ============================================================
#  reproduce_paper.sh — Full TH-GNN reproducibility pipeline
#
#  Given downloaded raw data, this script runs:
#    Stage 0  Sanity checks
#    Stage 1  Pre-compute RoBERTa embeddings (all 4 datasets)
#    Stage 2  Train LightGCN clean baselines
#    Stage 3  Full 720-experiment matrix (all attacks × detectors × seeds)
#    Stage 4  Ablation study (240 experiments)
#    Stage 5  Aggregate raw results → publication tables
#    Stage 6  Generate all analysis figures (notebooks/llm_vs_gan_analysis.ipynb)
#
#  All outputs land under results/.
#
#  Configurable via environment variables:
#    RAW_DIR     - raw data root       (default: data/raw)
#    CACHE_DIR   - processed data root (default: data/processed)
#    RESULTS_DIR - experiment outputs  (default: results/raw)
#    WORKERS     - parallel workers    (default: 1)
#    DEVICE      - torch device        (default: auto-detect: cuda > mps > cpu)
#    SKIP_EMBED  - skip Stage 1 if set (already have embeddings)
#    SKIP_BASELINE - skip Stage 2 if set
#
#  Usage:
#    bash scripts/reproduce_paper.sh                          # auto-detects device
#    WORKERS=4 DEVICE=cuda bash scripts/reproduce_paper.sh   # explicit GPU
#    DEVICE=cpu bash scripts/reproduce_paper.sh              # force CPU (smoke/debug only)
#    SKIP_EMBED=1 bash scripts/reproduce_paper.sh
#
#  NOTE: Stage 3 (720 experiments) is NOT feasible on CPU — run on a GPU machine.
#        On CPU, use --dry-run or restrict to a single dataset for testing.
# ============================================================

set -euo pipefail

# Parse flags
DRY_RUN=0
for _arg in "$@"; do
    [[ "$_arg" == "--dry-run" ]] && DRY_RUN=1
done

# ── Configurable paths ───────────────────────────────────────────────────────
RAW_DIR="${RAW_DIR:-data/raw}"
CACHE_DIR="${CACHE_DIR:-data/processed}"
RESULTS_DIR="${RESULTS_DIR:-results/raw}"
TABLES_DIR="${TABLES_DIR:-results/tables}"
FIGURES_DIR="${FIGURES_DIR:-results/figures}"
WORKERS="${WORKERS:-1}"
# Auto-detect device: respect $DEVICE env var if set; otherwise probe cuda > mps > cpu.
DEVICE=${DEVICE:-$(python3 -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
")}
DEVICE_FLAG="--device ${DEVICE}"

# ── Colour helpers ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
_die()   { echo -e "${RED}[FAIL]${NC}  $*" >&2; exit 1; }

# ── Timer helpers ───────────────────────────────────────────────────────────
_t0=""
_stage_start() { _t0=$SECONDS; _info "=== Stage $1: $2 ==="; }
_stage_end()   {
    local elapsed=$(( SECONDS - _t0 ))
    local m=$(( elapsed / 60 )); local s=$(( elapsed % 60 ))
    _ok "Stage $1 complete in ${m}m ${s}s"
}

# ── Stage 0: Sanity checks ───────────────────────────────────────────────────
_stage_start 0 "Sanity checks"

# Python ≥ 3.9
python_bin="${PYTHON:-python3}"
PY_VER=$($python_bin -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) \
    || _die "python3 not found. Install Python ≥ 3.9."
IFS='.' read -r PY_MAJOR PY_MINOR <<< "$PY_VER"
(( PY_MAJOR > 3 || (PY_MAJOR == 3 && PY_MINOR >= 9) )) \
    || _die "Python 3.9+ required, found $PY_VER."
_info "Python $PY_VER"

# Package installed
$python_bin -c "import src.models" 2>/dev/null \
    || _die "Package not installed. Run: pip install -e ."

# Required packages
for pkg in torch transformers sklearn numpy pandas scipy; do
    $python_bin -c "import $pkg" 2>/dev/null || _die "Missing package: $pkg. Run: pip install -r requirements.txt"
done

# Raw data dirs must exist
for ds in ml-1m amazon-books amazon-clothing yelp2018; do
    [[ -d "${RAW_DIR}/${ds}" ]] \
        || _warn "Raw data directory not found: ${RAW_DIR}/${ds}  (run scripts/download_data.py)"
done

# Create output dirs
mkdir -p "${CACHE_DIR}" "${RESULTS_DIR}" "${TABLES_DIR}" "${FIGURES_DIR}"

_info "Device: ${DEVICE}"
if [[ "${DEVICE}" == "cpu" ]]; then
    _warn "No GPU detected (device=cpu). Stages 1-4 will be slow."
    _warn "Stage 3 (720 experiments) is NOT feasible on CPU — consider a GPU machine."
fi

_stage_end 0

if [[ "$DRY_RUN" -eq 1 ]]; then
    _warn "Dry run: skipping Stages 1–6."
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  All done.  Results in results/${NC}"
    echo -e "${GREEN}============================================${NC}"
    exit 0
fi

# ── Stage 1: Pre-compute RoBERTa embeddings ─────────────────────────────────
if [[ -z "${SKIP_EMBED:-}" ]]; then
    _stage_start 1 "Pre-compute RoBERTa embeddings"
    $python_bin scripts/precompute_embeddings.py \
        --raw-dir       "${RAW_DIR}" \
        --processed-dir "${CACHE_DIR}" \
        --batch-size    64 \
        ${DEVICE_FLAG}
    _stage_end 1
else
    _warn "Stage 1 skipped (SKIP_EMBED set)"
fi

# ── Stage 2: Train LightGCN clean baselines ─────────────────────────────────
if [[ -z "${SKIP_BASELINE:-}" ]]; then
    _stage_start 2 "Train LightGCN clean baselines"
    $python_bin scripts/run_baseline.py \
        --raw-dir    "${RAW_DIR}" \
        --output     "${RESULTS_DIR}/../lightgcn_clean.json" \
        --epochs     100 \
        --lr         1e-3 \
        --embed-dim  64 \
        --n-layers   3 \
        --batch-size 2048 \
        --seed       42 \
        ${DEVICE_FLAG}
    _stage_end 2
else
    _warn "Stage 2 skipped (SKIP_BASELINE set)"
fi

# ── Stage 3: Full experiment matrix ─────────────────────────────────────────
_stage_start 3 "Full experiment matrix (720 experiments, workers=${WORKERS})"
$python_bin scripts/run_experiments.py \
    --raw-dir     "${RAW_DIR}" \
    --cache-dir   "${CACHE_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --workers     "${WORKERS}" \
    ${DEVICE_FLAG}
_stage_end 3

# ── Stage 4: Ablation study ─────────────────────────────────────────────────
_stage_start 4 "Ablation study (240 experiments)"
$python_bin scripts/run_ablation.py \
    --raw-dir     "${RAW_DIR}" \
    --cache-dir   "${CACHE_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --workers     "${WORKERS}" \
    ${DEVICE_FLAG}
_stage_end 4

# ── Stage 5: Aggregate → tables ─────────────────────────────────────────────
_stage_start 5 "Aggregate results → tables"
$python_bin scripts/aggregate_results.py \
    --results-dir "${RESULTS_DIR}" \
    --out-dir     "${TABLES_DIR}"
_stage_end 5

# ── Stage 6: Analysis figures (notebook) ────────────────────────────────────
_stage_start 6 "Generate analysis figures"
if $python_bin -c "import nbconvert, nbformat" 2>/dev/null; then
    $python_bin -m jupyter nbconvert \
        --to notebook \
        --execute \
        --ExecutePreprocessor.timeout=3600 \
        --output notebooks/llm_vs_gan_analysis_executed.ipynb \
        notebooks/llm_vs_gan_analysis.ipynb \
    && _ok "Executed notebook → notebooks/llm_vs_gan_analysis_executed.ipynb"
else
    _warn "nbconvert not installed — skipping notebook execution."
    _warn "Run manually: jupyter nbconvert --to notebook --execute notebooks/llm_vs_gan_analysis.ipynb"
fi
_stage_end 6

# ── Checksum manifest ───────────────────────────────────────────────────────
_info "Writing checksum manifest → results/checksums.sha256"
(
    cd "$(dirname "$0")/.."
    find "${TABLES_DIR}" -name "*.csv" -o -name "*.tex" | sort | \
        xargs shasum -a 256 2>/dev/null > results/checksums.sha256
    [[ -f "${RESULTS_DIR}/../lightgcn_clean.json" ]] && \
        shasum -a 256 "${RESULTS_DIR}/../lightgcn_clean.json" >> results/checksums.sha256
)
_ok "Checksums written."

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  All done.  Results in results/${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "  Tables  : ${TABLES_DIR}/"
echo "  LaTeX   : ${TABLES_DIR}/latex/"
echo "  Figures : ${FIGURES_DIR}/"
echo "  Checksums: results/checksums.sha256"
echo ""
