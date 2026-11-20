"""Plan part 9: one coherent daily Markdown report.

    report_schedule      the scheduled cutoff per reporting date, which date is due, early publication
    report_selection     eligible reviews: floor, seeds and withdrawals excluded, updates, labels, order
    revision_changes     the deterministic rule for "an update changes a result, assumption, or conclusion"
    report_snapshot      the frozen snapshot (selection, statistics, comparisons, corrections)
    report_comparisons   the saved comparisons a report shows
    report_synthesis     the writer role under its deadline and slot reservation, or the labelled fallback
    publication_lock     one publication process at a time
The command (commands/report_command.py) runs them in order; rendering is in rendering/daily_report.py.
"""
