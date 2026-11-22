"""`paperwatch backup DESTINATION` / `paperwatch backup --check-restore BACKUP_DIR`.

The first writes a consistent backup of the home (SQLite online backup API, settings, extracted text, output) into
DESTINATION/paperwatch-backup-<UTC time>/. The second checks a backup that was copied to a fresh directory: it loads
it as a paperwatch home and compares schema, integrity, row counts, and files with the backup's manifest. Neither
makes a network request."""

from __future__ import annotations

import argparse
import sys

from paperwatch.app_context import AppContext
from paperwatch.clock import utc_now
from paperwatch.maintenance.backup import BackupError, check_restored_home, create_backup

EXIT_CODE_BACKUP_FAILED = 14


def run_backup_command(app: AppContext, arguments: argparse.Namespace) -> int:
    if arguments.check_restore_dir is not None:
        restore_check = check_restored_home(arguments.check_restore_dir.expanduser().resolve())
        for problem in restore_check.problems:
            print(f"restore check: PROBLEM {problem}", file=sys.stderr)
        if restore_check.passed:
            print(
                f"restore check: {restore_check.restored_home} is a complete, consistent paperwatch home "
                f"(use it with --home {restore_check.restored_home})"
            )
        return 0 if restore_check.passed else EXIT_CODE_BACKUP_FAILED
    if arguments.destination_dir is None:
        print("paperwatch: FAILED [backup] give a DESTINATION directory or --check-restore BACKUP_DIR", file=sys.stderr)
        return EXIT_CODE_BACKUP_FAILED
    try:
        backup_result = create_backup(app.config, arguments.destination_dir, moment=utc_now())
    except (BackupError, OSError) as backup_failure:
        print(f"paperwatch: FAILED [backup] {backup_failure}", file=sys.stderr)
        return EXIT_CODE_BACKUP_FAILED
    print(backup_result.describe())
    return 0
