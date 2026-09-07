#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="${1:-advanced-ai-agent}"
VISIBILITY="${2:-private}"

if [[ "$VISIBILITY" != "private" && "$VISIBILITY" != "public" ]]; then
  echo "Visibility must be 'private' or 'public'." >&2
  exit 2
fi

command -v git >/dev/null 2>&1 || { echo "git is required." >&2; exit 1; }
command -v gh >/dev/null 2>&1 || { echo "GitHub CLI (gh) is required: https://cli.github.com/" >&2; exit 1; }

gh auth status >/dev/null

if [[ ! -d .git ]]; then
  git init -b main
fi

# Refuse to publish obvious secret files even if .gitignore was modified.
if git ls-files --cached --others --exclude-standard | grep -Eq '(^|/)(\.env$|.*\.pem$|.*\.key$|id_rsa$|id_ed25519$)'; then
  echo "Refusing to publish: a potential secret file is not ignored." >&2
  exit 3
fi

git add .
if ! git diff --cached --quiet; then
  git commit -m "Release Advanced AI Agent v2.5"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin HEAD:main
else
  gh repo create "$REPO_NAME" "--$VISIBILITY" --source=. --remote=origin --push
fi
