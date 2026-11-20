"""`paperwatch label-sample OUTPUT.csv [--size 40] [--seed N]` (plan part 10, ranking quality): write a fixed,
shuffled sample of screened candidates WITHOUT scores for the owner to mark "would read" and why. Read-only on the
database; never overwrites an existing file."""

from __future__ import annotations

import argparse
import sys

from paperwatch.app_context import AppContext
from paperwatch.measurement.label_sample import draw_label_sample, write_label_sheet

EXIT_CODE_LABEL_SAMPLE_FAILED = 15


def run_label_sample_command(app: AppContext, arguments: argparse.Namespace) -> int:
    sample = draw_label_sample(app.connection, app.config, sample_size=arguments.sample_size,
                               seed=arguments.sample_seed)
    if not sample:
        print("paperwatch: FAILED [label-sample] no screened candidates yet (run `collect` first)", file=sys.stderr)
        return EXIT_CODE_LABEL_SAMPLE_FAILED
    try:
        write_label_sheet(app.connection, app.config, sample, arguments.output_path)
    except FileExistsError:
        print(f"paperwatch: FAILED [label-sample] {arguments.output_path} exists; labels are never overwritten",
              file=sys.stderr)
        return EXIT_CODE_LABEL_SAMPLE_FAILED
    topic_count = len({candidate.topic_id for candidate in sample})
    print(f"wrote {len(sample)} candidate(s) across {topic_count} topic(s) to {arguments.output_path} (seed "
          f"{arguments.sample_seed}); fill in would_read (yes/no) and reason before looking at any score")
    return 0
