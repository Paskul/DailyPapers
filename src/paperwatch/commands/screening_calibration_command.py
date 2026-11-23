"""`paperwatch screening-calibration`: how a raw-score floor (screening.minimum_unnormalized_score) would have changed
which papers were read, judged by the topic assessor's verdicts, and the floor it recommends. Read-only on the
database; no network (paperwatch.measurement.screening_calibration)."""

from __future__ import annotations

import argparse

from paperwatch.app_context import AppContext
from paperwatch.measurement.screening_calibration import (
    calibrate_screening_floor,
    describe_screening_calibration,
    load_reviewed_paper_scores,
)


def run_screening_calibration_command(app: AppContext, arguments: argparse.Namespace) -> int:
    calibrations = calibrate_screening_floor(
        load_reviewed_paper_scores(app.connection), app.config.report.minimum_reading_priority
    )
    for line in describe_screening_calibration(app.config, calibrations):
        print(line)
    return 0
