#!/usr/bin/env bash
# Create the paperwatch service accounts and set directory permissions. Idempotent. Run with sudo.
#
# Usage: sudo deploy/setup_service_accounts.sh <paperwatch home> [operator account]
#
# Accounts and what they may touch:
#   paperwatch-collector  `collect`: reads config + provider credentials; writes state/ (SQLite) and state/cache/.
#                         Rootless Podman for the sandboxed extractor (gets subordinate ids and a lingering session).
#   paperwatch-worker     `work` + `report`: reads job inputs; calls the local model; writes state/ and output/.
#   paperwatch-state      shared group for state/, so both can use the SQLite database and its WAL files.
#   <operator>            (e.g. your desktop user) joins paperwatch-state to run `status` and edit config.
# llama-server keeps running as the operator for now (owner decision); see paperwatch-model-egress.nft for later.
#
# The paperwatch home must be readable by the service accounts: use a path such as /srv/paperwatch, not a
# directory inside a private /home/<user> (Ubuntu home directories are mode 0750).
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
[[ $# -ge 1 ]] || { echo "usage: sudo $0 <paperwatch home> [operator account]" >&2; exit 2; }
PAPERWATCH_HOME="$(realpath "$1")"
OPERATOR_ACCOUNT="${2:-${SUDO_USER:-}}"
COLLECTOR_ACCOUNT=paperwatch-collector
WORKER_ACCOUNT=paperwatch-worker
SHARED_GROUP=paperwatch-state
SUBORDINATE_ID_COUNT=65536

[[ -f "$PAPERWATCH_HOME/config.toml" ]] || { echo "$PAPERWATCH_HOME has no config.toml; run 'paperwatch --home $PAPERWATCH_HOME init' first" >&2; exit 1; }

# ---- accounts ----------------------------------------------------------------------------------------------------
getent group "$SHARED_GROUP" >/dev/null || groupadd --system "$SHARED_GROUP"
for ACCOUNT in "$COLLECTOR_ACCOUNT" "$WORKER_ACCOUNT"; do
    if ! id -u "$ACCOUNT" >/dev/null 2>&1; then
        useradd --system --user-group --groups "$SHARED_GROUP" --home-dir "/var/lib/$ACCOUNT" --create-home \
                --shell /usr/sbin/nologin "$ACCOUNT"
        echo "created account $ACCOUNT"
    else
        usermod --append --groups "$SHARED_GROUP" "$ACCOUNT"
    fi
    chmod 0750 "/var/lib/$ACCOUNT"
done
if [[ -n "$OPERATOR_ACCOUNT" ]]; then
    usermod --append --groups "$SHARED_GROUP" "$OPERATOR_ACCOUNT"
    echo "added $OPERATOR_ACCOUNT to $SHARED_GROUP (log out and in again for it to take effect)"
fi

# Rootless Podman for the collector: subordinate uid/gid ranges (system accounts get none by default) and a
# lingering user session so /run/user/<uid> exists for Podman's runtime files.
next_free_subordinate_id() {
    awk -F: -v minimum=200000 'BEGIN { next_id = minimum } { end = $2 + $3; if (end > next_id) next_id = end }
                               END { print next_id }' "$1"
}
if ! grep -q "^${COLLECTOR_ACCOUNT}:" /etc/subuid; then
    FIRST_ID=$(next_free_subordinate_id /etc/subuid)
    usermod --add-subuids "${FIRST_ID}-$((FIRST_ID + SUBORDINATE_ID_COUNT - 1))" "$COLLECTOR_ACCOUNT"
fi
if ! grep -q "^${COLLECTOR_ACCOUNT}:" /etc/subgid; then
    FIRST_ID=$(next_free_subordinate_id /etc/subgid)
    usermod --add-subgids "${FIRST_ID}-$((FIRST_ID + SUBORDINATE_ID_COUNT - 1))" "$COLLECTOR_ACCOUNT"
fi
loginctl enable-linger "$COLLECTOR_ACCOUNT"

# ---- directories and files ---------------------------------------------------------------------------------------
STATE_DIR="$PAPERWATCH_HOME/state"
CACHE_DIR="$STATE_DIR/cache"
OUTPUT_DIR="$PAPERWATCH_HOME/output"
CREDENTIALS_DIR="$PAPERWATCH_HOME/credentials"
LOCAL_SOURCES_DIR="$PAPERWATCH_HOME/local_sources"
mkdir -p "$STATE_DIR" "$CACHE_DIR" "$OUTPUT_DIR/papers" "$OUTPUT_DIR/daily" "$CREDENTIALS_DIR" "$LOCAL_SOURCES_DIR"

# Home and config: owned by the operator, readable (not writable) by the services.
chgrp "$SHARED_GROUP" "$PAPERWATCH_HOME" "$PAPERWATCH_HOME/config.toml" "$PAPERWATCH_HOME/topics.toml"
chmod 0750 "$PAPERWATCH_HOME"
chmod 0640 "$PAPERWATCH_HOME/config.toml" "$PAPERWATCH_HOME/topics.toml"

# state/: both services and the operator read and write SQLite and its -wal/-shm files (setgid keeps the group),
# and open the lock files (inference.lock, report-publication.lock) for writing.
chown "$WORKER_ACCOUNT:$SHARED_GROUP" "$STATE_DIR"
chmod 2770 "$STATE_DIR"
find "$STATE_DIR" -maxdepth 1 -type f \( -name '*.sqlite3*' -o -name '*.lock' \) \
     -exec chgrp "$SHARED_GROUP" {} + -exec chmod 0660 {} +

# state/cache/: written by the collector only, read by the worker.
chown -R "$COLLECTOR_ACCOUNT:$SHARED_GROUP" "$CACHE_DIR"
find "$CACHE_DIR" -type d -exec chmod 2750 {} +
find "$CACHE_DIR" -type f -exec chmod 0640 {} +

# local_sources/: PDFs/XML the operator places for papers (named by arXiv id or DOI); the collector only reads them.
chgrp -R "$SHARED_GROUP" "$LOCAL_SOURCES_DIR"
find "$LOCAL_SOURCES_DIR" -type d -exec chmod 2750 {} +
find "$LOCAL_SOURCES_DIR" -type f -exec chmod 0640 {} +

# output/: written by the worker (notes and reports) only, readable by the group.
chown -R "$WORKER_ACCOUNT:$SHARED_GROUP" "$OUTPUT_DIR"
find "$OUTPUT_DIR" -type d -exec chmod 2750 {} +
find "$OUTPUT_DIR" -type f -exec chmod 0640 {} +

# credentials/: the collector only. Nobody else can even list the directory.
chown -R "$COLLECTOR_ACCOUNT:$COLLECTOR_ACCOUNT" "$CREDENTIALS_DIR"
chmod 0700 "$CREDENTIALS_DIR"
find "$CREDENTIALS_DIR" -type f -exec chmod 0600 {} +

cat <<EOF

Accounts and permissions are set for $PAPERWATCH_HOME.
Run the services with a group-writable umask so new SQLite WAL files stay usable by both accounts, e.g.
  sudo -u $COLLECTOR_ACCOUNT sh -c 'umask 0007; exec paperwatch --home $PAPERWATCH_HOME collect'
  sudo -u $WORKER_ACCOUNT    sh -c 'umask 0007; exec paperwatch --home $PAPERWATCH_HOME work'
(the paperwatch systemd units set UMask=0007.)
To add provider keys, write them as the collector account:
  sudo -u $COLLECTOR_ACCOUNT sh -c 'umask 0077; \$EDITOR $CREDENTIALS_DIR/collector.toml'
Next: deploy/build_extractor_image.sh, then sudo deploy/apply_egress_rules.sh,
then sudo deploy/check_access_as_service_accounts.sh $PAPERWATCH_HOME
EOF
