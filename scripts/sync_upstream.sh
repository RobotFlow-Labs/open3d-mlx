#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

UPSTREAM_DIR="repositories/open3d-upstream"
UPSTREAM_REPO="https://github.com/isl-org/Open3D.git"

if [ ! -d "$UPSTREAM_DIR/.git" ]; then
    echo "Cloning upstream..."
    git clone --depth=1 "$UPSTREAM_REPO" "$UPSTREAM_DIR"
else
    echo "Updating upstream..."
    cd "$UPSTREAM_DIR"
    git fetch --depth=1 origin main
    git checkout FETCH_HEAD
    cd ../..
fi

HASH=$(cd "$UPSTREAM_DIR" && git rev-parse HEAD)
DATE=$(date -u +%Y-%m-%d)
cat > UPSTREAM_VERSION.md << EOF
# Upstream Reference

Synced to: **isl-org/Open3D** \`${HASH:0:12}\` (${DATE})

Repository: https://github.com/isl-org/Open3D
Commit: https://github.com/isl-org/Open3D/commit/${HASH}
EOF

echo "Synced to upstream: ${HASH:0:12}"
