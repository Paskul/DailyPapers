"""Comparisons of two completed reviews that address the same question (plan part 8).

    evidence_dimensions.py  which evidence dimensions two reviews share (assumptions, guarantees, setting)
    comparison_queue.py     queue at most one comparison per new review: its closest comparable neighbour on a
                            question both address, within the daily limit
    comparison_job.py       the worker's side: the comparison role's input from the two saved reviews, and the
                            stored comparison (keyed by question and the sorted pair of review ids: written once)
The comparison never rereads a paper: it reads the two validated reviews and the passages they already cite.
"""
