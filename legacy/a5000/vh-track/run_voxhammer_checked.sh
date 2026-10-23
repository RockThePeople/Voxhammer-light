#!/bin/bash
# D7: run VoxHammer with the failures its own code hides.
#
# WHY: voxhammer/extract_feature.py:125 catches every exception bare, prints
#   "Error happened during processing." and then the caller prints
#   "Feature extraction completed successfully!" and continues on a STALE
#   features.npz. W1 attempt 1 ran on July features and looked healthy.
#   VoxHammer source is not to be modified, so the guard lives out here.
#
# Guards:
#   PRE   free VRAM sufficient; refuses (never kills) if another process holds the GPU
#   POST  features.npz mtime must have advanced, else FAIL
#   POST  log must not contain the swallowed-error marker
#   POST  output .glb must exist and be newer than start
#
# Usage:
#   tools/run_voxhammer_checked.sh --input_model ... --mask_glb ... --output_dir ...
# Env:
#   VOXHAMMER_DIR      default <HOME>/ai-ar-v2/VoxHammer
#   REQUIRE_FREE_MIB   default 16000   (W1 measured 14180 MiB peak)
#   RENDER_DIR         default <output_dir>/render
#   VOXHAMMER_CMD      override the run command (test hook only)
set -uo pipefail

VOXHAMMER_DIR="${VOXHAMMER_DIR:-<HOME>/ai-ar-v2/VoxHammer}"
REQUIRE_FREE_MIB="${REQUIRE_FREE_MIB:-16000}"
MARKER="Error happened during processing."

die() { echo "FAIL: $*" >&2; exit 1; }
ok()  { echo "  ok: $*"; }

# ---- parse just enough to locate the artifacts we must verify ----
OUTPUT_DIR=outputs
for ((i=1;i<=$#;i++)); do
  if [ "${!i}" = "--output_dir" ]; then j=$((i+1)); OUTPUT_DIR="${!j}"; fi
  if [ "${!i}" = "--render_dir" ]; then j=$((i+1)); RENDER_DIR_ARG="${!j}"; fi
done
cd "$VOXHAMMER_DIR" || die "no VoxHammer dir: $VOXHAMMER_DIR"
RENDER_DIR="${RENDER_DIR:-${RENDER_DIR_ARG:-$OUTPUT_DIR/render}}"
FEAT="$RENDER_DIR/features.npz"
GLB="${OUTPUT_GLB:-$OUTPUT_DIR/output.glb}"
# Round-trip / decode-only runs never invoke extract_feature.py, so demanding a
# fresh features.npz would be a false FAIL. Strict by default; opt out explicitly.
EXPECT_FEATURES_REGEN="${EXPECT_FEATURES_REGEN:-1}"

echo "== PRE-FLIGHT =="

# --- GPU: refuse, never kill. Other sessions may legitimately own the card. ---
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
[ -z "$FREE" ] && die "nvidia-smi unavailable"
APPS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
if [ -n "$APPS" ]; then
  echo "  other GPU processes present:"; echo "$APPS" | sed 's/^/    /'
fi
if [ "$FREE" -lt "$REQUIRE_FREE_MIB" ]; then
  echo "$APPS" | sed 's/^/    /' >&2
  die "only ${FREE} MiB free, need ${REQUIRE_FREE_MIB}. Refusing (will not kill foreign processes)."
fi
ok "VRAM free ${FREE} MiB >= ${REQUIRE_FREE_MIB}"

# --- record pre-state for the post-checks ---
PRE_FEAT_MTIME=0
[ -f "$FEAT" ] && PRE_FEAT_MTIME=$(stat -c %Y "$FEAT")
if [ "$PRE_FEAT_MTIME" != 0 ]; then
  ok "features.npz pre-mtime $PRE_FEAT_MTIME ($(date -d @$PRE_FEAT_MTIME '+%F %T'))"
else
  ok "features.npz absent (first run)"
fi
START=$(date +%s)
LOG="${LOG:-$OUTPUT_DIR/voxhammer_run_$START.log}"
mkdir -p "$(dirname "$LOG")"

echo "== RUN =="
# env.sh appends to LD_LIBRARY_PATH/CPLUS_INCLUDE_PATH which may be unset;
# relax -u only across the source. It is VoxHammer's file -- do not edit it.
if [ -f ./env.sh ]; then set +u; source ./env.sh; set -u; fi
# 로그 각 줄에 **시작 이후 경과초**를 찍는다. 단계별 소요를 사후에 쪼개려면 이게 있어야 한다
# (inference.py 는 단계 이름만 찍고 시간을 안 찍는다). PIPESTATUS 로 파이썬 종료코드를 살린다.
STAMP='import sys,time
t0=time.time()
for line in sys.stdin:
    sys.stdout.write("[%8.2f] %s" % (time.time()-t0, line)); sys.stdout.flush()'
if [ -n "${VOXHAMMER_CMD:-}" ]; then
  bash -c "$VOXHAMMER_CMD" 2>&1 | ./.venv/bin/python -c "$STAMP" > "$LOG"; RC=${PIPESTATUS[0]}
else
  ./.venv/bin/python -u inference.py "$@" 2>&1 | ./.venv/bin/python -c "$STAMP" > "$LOG"; RC=${PIPESTATUS[0]}
fi
echo "  exit=$RC  log=$LOG  wall=$(( $(date +%s) - START ))s"

echo "== POST-FLIGHT =="
FAILED=0

# 1. the swallowed error
if grep -qF "$MARKER" "$LOG"; then
  echo "FAIL: log contains swallowed error: '$MARKER'" >&2
  echo "      VoxHammer prints 'completed successfully' anyway -- do not trust this run." >&2
  FAILED=1
else
  ok "no swallowed-error marker in log"
fi

# 2. features.npz must be regenerated  <-- the check W1 lacked
if [ "$EXPECT_FEATURES_REGEN" = "0" ]; then
  ok "features.npz regen check SKIPPED (EXPECT_FEATURES_REGEN=0; run does no feature extraction)"
elif [ ! -f "$FEAT" ]; then
  echo "FAIL: $FEAT missing" >&2; FAILED=1
else
  POST=$(stat -c %Y "$FEAT")
  if [ "$POST" -le "$PRE_FEAT_MTIME" ]; then
    echo "FAIL: features.npz NOT regenerated (mtime $POST <= pre $PRE_FEAT_MTIME)" >&2
    echo "      => feature extraction silently fell back to a stale cache." >&2
    FAILED=1
  else
    ok "features.npz regenerated (mtime $PRE_FEAT_MTIME -> $POST)"
  fi
fi

# 3. output produced by THIS run
if [ ! -f "$GLB" ]; then
  echo "FAIL: $GLB missing" >&2; FAILED=1
elif [ "$(stat -c %Y "$GLB")" -lt "$START" ]; then
  echo "FAIL: $GLB is stale (predates this run)" >&2; FAILED=1
else
  ok "$GLB produced by this run"
fi

# 4. the process's own exit code
[ "$RC" -ne 0 ] && { echo "FAIL: inference.py exit $RC" >&2; FAILED=1; }

[ "$FAILED" -ne 0 ] && { echo "RESULT: FAIL" >&2; exit 1; }
echo "RESULT: PASS"
