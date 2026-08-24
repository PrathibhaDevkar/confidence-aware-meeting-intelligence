#!/usr/bin/env bash
# Downloads QMSum (GitHub) and the text-only MeetingBank dataset (Hugging Face)
# into data/raw/. Re-run is safe: existing downloads are skipped.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT_DIR/data/raw"
PYTHON="$ROOT_DIR/venv/bin/python3"
cd "$ROOT_DIR"

mkdir -p "$RAW_DIR"

# --- QMSum (~26 MB zip, ~98 MB unpacked) ---
if [ -d "$RAW_DIR/qmsum/QMSum-main" ]; then
  echo "QMSum already present, skipping."
else
  echo "Downloading QMSum..."
  mkdir -p "$RAW_DIR/qmsum"
  curl -L -o "$RAW_DIR/qmsum/qmsum.zip" \
    "https://github.com/Yale-LILY/QMSum/archive/refs/heads/main.zip"
  unzip -q "$RAW_DIR/qmsum/qmsum.zip" -d "$RAW_DIR/qmsum"
  rm "$RAW_DIR/qmsum/qmsum.zip"
  echo "QMSum downloaded to $RAW_DIR/qmsum/QMSum-main"
fi

# --- MeetingBank, text-only (~115 MB via Hugging Face `datasets`) ---
if [ -d "$RAW_DIR/meetingbank" ] && [ "$(ls -A "$RAW_DIR/meetingbank" 2>/dev/null)" ]; then
  echo "MeetingBank already present, skipping."
else
  echo "Downloading MeetingBank (text-only, ~115 MB)..."
  mkdir -p "$RAW_DIR/meetingbank"
  "$PYTHON" - <<'PYEOF'
from datasets import load_dataset

ds = load_dataset("huuuyeah/meetingbank")
for split in ds:
    ds[split].to_json(f"data/raw/meetingbank/{split}.jsonl")
    print(f"{split}: {len(ds[split])} rows -> data/raw/meetingbank/{split}.jsonl")
PYEOF
fi

echo "Done."
