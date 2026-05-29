#!/usr/bin/env bash
# =============================================================================
# RandOpt speedrun — Prime Intellect single-GPU smoke test
#
# Provisions the CHEAPEST available single-GPU pod, copies this repo, validates
# the Triton kernel against the torch reference on real CUDA (tier 1), then runs
# the end-to-end smoke speedrun (tier 2, best-effort), copies the record back,
# and ALWAYS terminates the pod (EXIT trap).
#
# Safety: dry-run by default. It prints the chosen offer + hourly price and EXITS
# unless you pass CONFIRM=1. The pod is torn down on any exit (success/failure/^C).
#
# Usage:
#   bash scripts/prime_smoke.sh                      # dry run: show cheapest offer
#   CONFIRM=1 bash scripts/prime_smoke.sh            # provision + run + teardown
#   CONFIRM=1 GPU_TYPE=A100_80GB IMAGE=... bash scripts/prime_smoke.sh
#
# Requires: a VALID PRIME_API_KEY in ./env.local, the `prime` CLI, ssh/rsync.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- knobs ------------------------------------------------------------------
GPU_TYPE=${GPU_TYPE:-}                         # empty = cheapest of any 1-GPU offer
GPU_COUNT=${GPU_COUNT:-1}
DISK=${DISK:-100}
IMAGE=${IMAGE:-}                               # e.g. a CUDA+PyTorch image; empty=provider default
POD_NAME=${POD_NAME:-randopt-smoke-$(date +%s)}
CONFIG=${CONFIG:-configs/smoke_1gpu_small.yaml}
PRIME=${PRIME:-prime}                          # or ".venv/bin/prime"
CONFIRM=${CONFIRM:-0}
REMOTE_DIR=/root/RandOpt

redact(){ sed -E 's/pit_[A-Za-z0-9]+/pit_***REDACTED***/g'; }

# ---- 0. auth ----------------------------------------------------------------
[ -f env.local ] || { echo "env.local with PRIME_API_KEY required"; exit 1; }
set -a; . ./env.local; set +a
: "${PRIME_API_KEY:?PRIME_API_KEY not set in env.local}"
$PRIME config set-api-key "$PRIME_API_KEY" >/dev/null 2>&1 || true
if ! $PRIME whoami --plain >/dev/null 2>&1; then
  echo "ERROR: prime auth failed (key unauthorized/expired). Generate a new key at"
  echo "       https://app.primeintellect.ai/dashboard/tokens and put it in env.local"
  exit 1
fi

# ---- 1. pick the cheapest available offer -----------------------------------
echo ">> querying availability..."
OFFERS_JSON=$($PRIME availability list ${GPU_TYPE:+--gpu-type "$GPU_TYPE"} --gpu-count "$GPU_COUNT" -o json)
read -r OFFER_ID OFFER_PRICE OFFER_GPU OFFER_SOCKET OFFER_PROVIDER OFFER_GPUMEM <<EOF
$(python3 - "$OFFERS_JSON" <<'PY'
import json, sys
data = json.loads(sys.argv[1])
res = [r for r in data.get("gpu_resources", []) if str(r.get("stock_status","")).lower() in ("available","high","medium","low","yes","true","in_stock") or r.get("price_value") is not None]
res = [r for r in res if r.get("price_value") is not None]
if not res:
    print(""); sys.exit(0)
best = min(res, key=lambda r: r["price_value"])
print(best.get("id",""), best.get("price_per_hour", best.get("price_value")),
      best.get("gpu_type",""), best.get("socket",""), best.get("provider",""), best.get("gpu_memory",""))
PY
)
EOF
[ -n "${OFFER_ID:-}" ] || { echo "No available single-GPU offers found for GPU_TYPE='${GPU_TYPE:-any}'."; exit 1; }
echo ">> cheapest offer: id=$OFFER_ID  $OFFER_GPU x$GPU_COUNT  ${OFFER_PRICE}/hr  (${OFFER_PROVIDER}/${OFFER_SOCKET}, ${OFFER_GPUMEM} mem)"

# ---- 2. confirm (dry run unless CONFIRM=1) ----------------------------------
if [ "$CONFIRM" != "1" ]; then
  echo ">> DRY RUN. Re-run with CONFIRM=1 to provision this pod and run the smoke."
  exit 0
fi

# ---- 3. create pod + ALWAYS terminate on exit -------------------------------
POD_ID=""
cleanup(){ if [ -n "$POD_ID" ]; then echo ">> terminating pod $POD_ID"; $PRIME pods terminate "$POD_ID" -y >/dev/null 2>&1 || true; fi; }
trap cleanup EXIT INT TERM

echo ">> creating pod '$POD_NAME'..."
$PRIME pods create --id "$OFFER_ID" --name "$POD_NAME" --gpu-count "$GPU_COUNT" \
    --disk-size "$DISK" ${IMAGE:+--image "$IMAGE"} -y --plain | redact

# resolve pod id by name
for _ in $(seq 1 30); do
  POD_ID=$($PRIME pods list -o json 2>/dev/null | python3 -c "import sys,json;
ps=json.load(sys.stdin); ps=ps if isinstance(ps,list) else ps.get('pods',ps.get('data',[]));
print(next((p['id'] for p in ps if p.get('name')=='$POD_NAME'),''))" 2>/dev/null || true)
  [ -n "$POD_ID" ] && break; sleep 3
done
[ -n "$POD_ID" ] || { echo "could not resolve pod id"; exit 1; }
echo ">> pod id: $POD_ID"

# ---- 4. wait for ACTIVE + ssh -----------------------------------------------
echo ">> waiting for pod to become active..."
SSH_STR=""
for _ in $(seq 1 100); do
  ST_JSON=$($PRIME pods status "$POD_ID" -o json 2>/dev/null || echo '{}')
  read -r STATUS SSH_STR <<EOF
$(python3 - "$ST_JSON" <<'PY'
import json,sys
d=json.loads(sys.argv[1] or "{}")
print(d.get("status",""), d.get("ssh","") or "")
PY
)
EOF
  echo "   status=$STATUS"
  { [ "$STATUS" = "ACTIVE" ] || [ "$STATUS" = "RUNNING" ]; } && [ -n "$SSH_STR" ] && break
  sleep 6
done
[ -n "$SSH_STR" ] || { echo "pod never exposed ssh"; exit 1; }
echo ">> ssh: $(echo "$SSH_STR" | redact)"

# parse "ssh user@host -p PORT" -> user/host/port
read -r SSH_USERHOST SSH_PORT <<EOF
$(python3 - "$SSH_STR" <<'PY'
import re,sys
s=sys.argv[1]
m=re.search(r'(\S+@\S+)',s); uh=m.group(1) if m else s
p=re.search(r'-p\s*(\d+)',s); print(uh, p.group(1) if p else "22")
PY
)
EOF
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT"
run_remote(){ ssh $SSH_OPTS "$SSH_USERHOST" "$1"; }

echo ">> waiting for sshd..."
for _ in $(seq 1 30); do run_remote "echo ok" >/dev/null 2>&1 && break; sleep 5; done

# ---- 5. copy repo subset (skip .venv/.git/baselines/data) -------------------
echo ">> syncing repo..."
run_remote "mkdir -p $REMOTE_DIR"
rsync -az -e "ssh $SSH_OPTS" \
  --exclude '.venv' --exclude '.git' --exclude 'baselines' --exclude 'speedrun-runs' \
  --exclude '__pycache__' --exclude '.pytest_cache' --exclude 'env.local' \
  ./ "$SSH_USERHOST:$REMOTE_DIR/"

# ---- 6. tier 1: kernel numerics (torch+triton only) -------------------------
echo ">> [tier1] installing torch+triton+pytest and validating the Triton kernel..."
run_remote "cd $REMOTE_DIR && pip install -q --upgrade pip && pip install -q pytest numpy 'torch' triton && python -m pytest tests/test_kernel_gpu.py -q -s"

# ---- 7. tier 2: end-to-end smoke (vllm + data) [best-effort] -----------------
echo ">> [tier2] installing full deps + gsm8k data + running speedrun smoke (best-effort)..."
run_remote "cd $REMOTE_DIR && pip install -q -r requirements.txt && python scripts/make_gsm8k_smoke.py --n-train 32 --n-test 32 && python speedrun.py --config $CONFIG" || echo "   (tier2 e2e smoke failed — kernel numerics in tier1 are the critical check)"

# ---- 8. fetch records -------------------------------------------------------
echo ">> fetching records..."
mkdir -p speedrun-runs
rsync -az -e "ssh $SSH_OPTS" "$SSH_USERHOST:$REMOTE_DIR/speedrun-runs/" speedrun-runs/ 2>/dev/null || true
rsync -az -e "ssh $SSH_OPTS" "$SSH_USERHOST:$REMOTE_DIR/RECORDS.md" ./RECORDS.smoke.md 2>/dev/null || true

echo ">> done. Pod will be terminated by the EXIT trap."
