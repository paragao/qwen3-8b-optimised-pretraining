#!/usr/bin/env bash
# Scan this repo's shell scripts for constructs that are NOT available in
# bash 3.2 -- the version macOS ships as /bin/bash, and therefore the version
# a laptop-side driver script actually runs under.
#
# WHY THIS EXISTS
# `bash -n` cannot catch any of these: a missing builtin is a runtime lookup
# failure, not a syntax error. g5/finish-run-2node.sh shipped with seven
# `mapfile` calls and one `wait -n`, and the first of them failed only after
# the script had already launched two instances and opened SSH sessions --
# i.e. after it had started costing money.
#
# Usage:  ./g5/lint-portability.sh            # scan, exit 1 on any finding
#         ./g5/lint-portability.sh --list     # just list the files scanned
set -euo pipefail
cd "$(dirname "$0")/.."

FILES=$(find g5 -name '*.sh' -type f | sort)

if [[ "${1:-}" == "--list" ]]; then
  echo "${FILES}"
  exit 0
fi

echo "bash on this host: $(bash --version | head -1)"
echo "scanning $(echo "${FILES}" | wc -l | tr -d ' ') shell scripts under g5/"
echo

findings=0

# Each rule is "pattern<TAB>explanation". Patterns are extended regexes using
# POSIX character classes -- NOT \s, which BSD/macOS grep -E does not support.
report() {   # pattern  explanation
  local pat="$1" why="$2" hits
  # Exclude this file (it necessarily contains the patterns) and comment lines.
  hits=$(grep -nE "${pat}" ${FILES} 2>/dev/null \
           | grep -v '^g5/lint-portability.sh:' \
           | grep -vE ':[0-9]+:[[:space:]]*#' || true)
  if [[ -n "${hits}" ]]; then
    echo "FAIL  ${why}"
    echo "${hits}" | sed 's/^/        /'
    echo
    findings=$(( findings + 1 ))
  else
    echo "ok    ${why}"
  fi
}

report '(^|[^[:alnum:]_])(mapfile|readarray)[[:space:]]' \
  'mapfile/readarray are bash 4.0+ (use a read loop fed by process substitution)'

report '(^|[^[:alnum:]_])wait[[:space:]]+-' \
  'wait with any flag (-n, -f, -p) is bash 4.3+ (wait on explicit PIDs instead)'

report '(^|[^[:alnum:]_])wait([[:space:]]*$|[[:space:]]*\|\|)' \
  'bare wait returns 0 even when a job FAILED, so it cannot detect failure'

report '(declare|typeset|local)[[:space:]]+-[A-Za-z]*A' \
  'associative arrays (declare -A) are bash 4.0+'

report '(declare|typeset|local)[[:space:]]+-[A-Za-z]*n[[:space:]]' \
  'namerefs (local -n / declare -n) are bash 4.3+'

report '\$\{[A-Za-z_][A-Za-z0-9_]*(,,|\^\^)' \
  'case conversion ${v,,} / ${v^^} is bash 4.0+ (use tr)'

report '(^|[^[:alnum:]_])coproc[[:space:]]' \
  'coproc is bash 4.0+'

report 'globstar' \
  'shopt -s globstar is bash 4.0+'

report '(grep|sed)[^|;]*-E[^|;]*\\s' \
  'backslash-s in grep -E / sed -E silently matches nothing on BSD/macOS'

echo
if [[ "${findings}" -ne 0 ]]; then
  echo "RESULT: ${findings} rule(s) failed -- these break on macOS /bin/bash 3.2."
  exit 1
fi
echo "RESULT: clean -- every scanned script should run on bash 3.2."
