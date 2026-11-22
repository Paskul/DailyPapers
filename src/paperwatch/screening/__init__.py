"""Score every candidate cheaply on the CPU, then promote the best ones to a reading, in order.

Modules, in the order one `collect` run uses them (screening_stage.py drives them):
    embedding_model_registry  the pinned sentence-embedding models: files, sha256 hashes, pooling, query instruction
    text_embedders            the in-process ONNX embedder (CPU) and a word-hashing stand-in for offline tests
    embedding_cache           vectors cached in SQLite by model identity and text hash
    phrase_matching           the lexical signal: approved phrases found verbatim (catches exact terms like "Dafny")
    topic_screening_profiles  what each topic is scored against: question text, active seeds, phrases, fingerprint
    screening_score           the pure scoring formula: three signals, documented normalization, renormalized weights
    signal_normalization      each signal as its percentile within the topic's pool, so signals and topics compare
    candidate_screening       which candidates need a (re)score, batch embedding, and storing the scores
    promotion_selection       the pure slot rule: topic shares first, spare slots by score, a few near the cutoff
    review_promotion          promotion in the database: the daily limit, queueing the reading in its priority class
Ordinary Python code owns all of it; no model call is made, and nothing leaves the machine.
"""
