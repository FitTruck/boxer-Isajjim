#!/usr/bin/env bash
# One-time install: point git at scripts/hooks/ for project-shared hooks.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

git config core.hooksPath scripts/hooks
chmod +x scripts/hooks/pre-commit

echo "✓ git core.hooksPath → scripts/hooks"
echo "✓ pre-commit hook is now active for this clone."
echo ""
echo "Bypass once:   SKIP_DOC_SYNC=1 git commit ..."
echo "Uninstall:     git config --unset core.hooksPath"
