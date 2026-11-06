#!/usr/bin/env bash
# Compatibility wrapper: the canonical manager lives one level up.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../manage_projects.sh" "$@"
