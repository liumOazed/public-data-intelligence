#!/usr/bin/env bash
# Open a GitHub Issue for a failed scheduled run, or add a comment if one is already open.
# Usage: open_issue.sh "<title>" "<body>"
set -euo pipefail
title="$1"
body="$2"
existing=$(gh issue list --state open --search "in:title \"$title\"" --json number,title \
  --jq "map(select(.title == \"$title\")) | .[0].number // empty")
if [ -n "$existing" ]; then
  gh issue comment "$existing" --body "$body"
else
  gh issue create --title "$title" --body "$body"
fi
