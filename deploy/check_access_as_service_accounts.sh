#!/usr/bin/env bash
# Run `paperwatch access-check` as each service account (plan part 3: test the rules from the actual accounts).
#
# Usage: sudo deploy/check_access_as_service_accounts.sh <paperwatch home>
# Exit status 0 only if both accounts pass every required check.
set -uo pipefail

[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
[[ $# -ge 1 ]] || { echo "usage: sudo $0 <paperwatch home>" >&2; exit 2; }
PAPERWATCH_HOME="$(realpath "$1")"
PAPERWATCH_BIN="${PAPERWATCH_BIN:-/opt/paperwatch/venv/bin/paperwatch}"

# Resolve the probe host here, as root, so the worker's check can also try a direct connection to a literal IPv4
# and IPv6 address: blocking name resolution alone would not prove the egress rules.
PROBE_ADDRESS_ARGUMENTS=()
IPV4_PROBE_ADDRESS="$(getent ahostsv4 export.arxiv.org | awk 'NR == 1 { print $1 }')"
IPV6_PROBE_ADDRESS=""
for PROBE_HOST in export.arxiv.org api.openalex.org; do  # provider hosts only; arXiv has no IPv6 address
    IPV6_PROBE_ADDRESS="$(getent ahostsv6 "$PROBE_HOST" | awk '$1 !~ /^::ffff:/ { print $1; exit }')"
    [[ -n "$IPV6_PROBE_ADDRESS" ]] && break
done
for PROBE_ADDRESS in $IPV4_PROBE_ADDRESS $IPV6_PROBE_ADDRESS; do
    PROBE_ADDRESS_ARGUMENTS+=(--public-probe-address "$PROBE_ADDRESS")
done

overall_status=0
for ACCOUNT_AND_COMPONENT in "paperwatch-collector collector" "paperwatch-worker worker"; do
    read -r ACCOUNT COMPONENT <<<"$ACCOUNT_AND_COMPONENT"
    echo "=== $ACCOUNT ($COMPONENT) ==="
    ACCOUNT_UID="$(id -u "$ACCOUNT")" || { overall_status=1; continue; }
    # cd / : the service account cannot read the caller's working directory.
    COMPONENT_ARGUMENTS=()
    if [[ "$COMPONENT" == worker ]]; then COMPONENT_ARGUMENTS=("${PROBE_ADDRESS_ARGUMENTS[@]}"); fi
    (cd / && sudo -u "$ACCOUNT" -H env "XDG_RUNTIME_DIR=/run/user/$ACCOUNT_UID" \
        "$PAPERWATCH_BIN" --home "$PAPERWATCH_HOME" access-check --component "$COMPONENT" \
        "${COMPONENT_ARGUMENTS[@]}") || overall_status=1
    echo
done
exit $overall_status
