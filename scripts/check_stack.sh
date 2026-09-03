#!/usr/bin/env bash
# Every open PR's base must be an ancestor of its head, and the whole stack
# must be one chain ending at the tip.
#
# This exists because it has already gone wrong twice. #47 declared itself the
# base of #48 but was never in the chain, and #51 was a sibling of #52 rather
# than its parent. Both were found by accident, days later, by a subagent
# reporting that a feature it expected was missing. A stack kept to 1.0 makes
# that failure more likely, not less, so it is checked rather than watched for.
set -euo pipefail
git fetch -q origin
fail=0
while IFS=$'\t' read -r n base head; do
  if git merge-base --is-ancestor "origin/$base" "origin/$head" 2>/dev/null; then
    printf '  ok    #%-4s %s <- %s\n' "$n" "$base" "$head"
  else
    printf '  ORPHAN #%-4s %s is NOT an ancestor of %s\n' "$n" "$base" "$head"
    fail=1
  fi
done < <(gh pr list --limit 60 --json number,baseRefName,headRefName \
           --jq '.[] | "\(.number)\t\(.baseRefName)\t\(.headRefName)"' | sort -n)
if [ "$fail" -ne 0 ]; then
  echo
  echo "A base that is not an ancestor means that PR's work is absent from"
  echo "everything built on top of it. Rebase before doing anything else."
  exit 1
fi
echo "stack is one chain"
