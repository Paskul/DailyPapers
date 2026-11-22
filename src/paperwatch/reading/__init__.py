"""Read the selected papers and save evidence-backed notes.

One reading, in the worker:
  review_packet.py       the edition-specific packet: passages with stable ids and pages/sections (passages.py),
                         the reading plan (reading_plan.py), and parser recovery (parser_recovery.py)
  reading_pipeline.py    section reader calls (saved one by one, stage_results.py), the evidence synthesis, and the
                         topic assessment with comparison anchors (comparison_anchors.py)
  evidence_checks.py     passage ids, excerpts, and numbers are checked against the saved edition before commit
  reading_priority.py    the 0-100 rubric; the model proposes ratings, this code computes the score
The model roles themselves live in paperwatch.roles (section_reader, reviewer, topic_assessor).
"""
