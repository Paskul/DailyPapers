"""paperwatch: a local reading service that answers "Which papers are worth my time today, and why?"

Ordinary Python code owns network requests, scheduling, scores, database changes, and file writes.
The local llama.cpp model only ever receives plain data and returns schema-validated JSON.
"""

import os

__version__ = "0.1.0"

# numpy's bundled OpenBLAS starts one thread per CPU core, and idle ones busy-wait after each call. paperwatch's numpy
# work is small vector arithmetic, so one thread is just as fast: on a 128-thread machine the test suite used 13x
# less CPU in the same time. (Screening's CPU time is the ONNX embedding model's own threads,
# screening.embedding_threads, and is unchanged.) Set before any paperwatch module imports numpy, which reads it
# once; an operator's own setting wins.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
