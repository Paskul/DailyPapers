#!/usr/bin/env bash
# Render, check, install, and load the paperwatch nftables egress rules. Run with sudo.
#
# Usage: sudo deploy/apply_egress_rules.sh [--model-port 8080] [--with-model-account ACCOUNT]
#
# The service accounts must exist first (deploy/setup_service_accounts.sh): nft resolves account names at load time.
# Rules are installed to /etc/paperwatch/nftables/, loaded immediately, and kept across reboots by the dedicated
# paperwatch-egress.service. Do NOT use the stock nftables.service for this: its /etc/nftables.conf starts with
# `flush ruleset`, which would also wipe UFW's or Docker's rules.
set -euo pipefail

MODEL_PORT=8080
MODEL_ACCOUNT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-port) MODEL_PORT="$2"; shift 2 ;;
        --with-model-account) MODEL_ACCOUNT="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
[[ "$MODEL_PORT" =~ ^[0-9]{1,5}$ ]] || { echo "--model-port must be a number" >&2; exit 2; }
for ACCOUNT in paperwatch-worker paperwatch-collector ${MODEL_ACCOUNT}; do
    id -u "$ACCOUNT" >/dev/null 2>&1 || { echo "account $ACCOUNT does not exist; run setup_service_accounts.sh" >&2; exit 1; }
done

SOURCE_DIR="$(cd "$(dirname "$0")/nftables" && pwd)"
INSTALL_DIR=/etc/paperwatch/nftables
install -d -m 0755 "$INSTALL_DIR"

sed "s/@MODEL_PORT@/${MODEL_PORT}/g" "$SOURCE_DIR/paperwatch-egress.nft" > "$INSTALL_DIR/paperwatch-egress.nft"
RULE_FILES=("$INSTALL_DIR/paperwatch-egress.nft")
if [[ -n "$MODEL_ACCOUNT" ]]; then
    sed "s/@MODEL_ACCOUNT@/${MODEL_ACCOUNT}/g" "$SOURCE_DIR/paperwatch-model-egress.nft" \
        > "$INSTALL_DIR/paperwatch-model-egress.nft"
    RULE_FILES+=("$INSTALL_DIR/paperwatch-model-egress.nft")
else
    rm -f "$INSTALL_DIR/paperwatch-model-egress.nft"
    nft list table inet paperwatch_model_egress >/dev/null 2>&1 && nft delete table inet paperwatch_model_egress
fi
chmod 0644 "${RULE_FILES[@]}"

for RULE_FILE in "${RULE_FILES[@]}"; do
    nft --check --file "$RULE_FILE"
    nft --file "$RULE_FILE"
    echo "loaded $RULE_FILE"
done

# Persistence: a oneshot unit that loads only our files at boot (never flushes other rule sets).
install -m 0644 "$(cd "$(dirname "$0")/systemd" && pwd)/paperwatch-egress.service" \
    /etc/systemd/system/paperwatch-egress.service
systemctl daemon-reload
systemctl enable paperwatch-egress.service
systemctl restart paperwatch-egress.service
echo "paperwatch-egress.service enabled: the rules are reloaded at every boot"

cat <<EOF

Verify from the service accounts:
  sudo deploy/check_access_as_service_accounts.sh <paperwatch home>
EOF
