#!/usr/bin/env bash
# Downloads the real MagnaTagATune audio (16kHz/32kbps mono mp3, split across
# 3 x ~1GB zip parts) so the `path` column in the LP-MusicCaps-MTT parquet
# files resolves to real mp3s. Official host: City University MIRG
# (https://mirg.city.ac.uk/datasets/magnatagatune/index1.html), mirroring
# the dataset since tagatune.org went offline. Needs real internet access
# and ~3GB+ disk -- run on your machine or Colab, not in this repo's
# scaffolding sandbox.
#
# Usage:
#   bash scripts/download_audio.sh [DEST]
set -euo pipefail

DEST="${1:-data/raw/audio}"
mkdir -p "$DEST"
cd "$DEST"

BASE="https://mirg.city.ac.uk/datasets/magnatagatune"
official_ok=true
for part in 001 002 003; do
  if [ ! -f "mp3.zip.${part}" ] && ! curl --fail --retry 3 -L -O "${BASE}/mp3.zip.${part}"; then
    official_ok=false
    break
  fi
done

# The official server is intermittently unavailable. Hugging Face hosts a
# byte-equivalent consolidated archive used by the confit/magnatagatune
# dataset loader, so use it as a reproducible fallback.
if [ "$official_ok" = false ]; then
  rm -f mp3.zip.001 mp3.zip.002 mp3.zip.003
  curl --fail --retry 3 -L \
    "https://huggingface.co/datasets/confit/magnatagatune/resolve/main/mp3.zip" \
    -o mp3_full.zip
else
  cat mp3.zip.001 mp3.zip.002 mp3.zip.003 > mp3_full.zip
fi

unzip -q mp3_full.zip
rm -f mp3_full.zip

echo "MagnaTagATune audio extracted into $DEST"
echo "This gives numbered folders (0/ .. f/) matching the 'path' column in"
echo "data/lp_musiccaps/{train,valid,test}.parquet, e.g."
echo "  $DEST/0/american_bach_soloists-j_s__bach__cantatas_volume_v-...-0-29.mp3"
