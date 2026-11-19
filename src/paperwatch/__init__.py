"""paperwatch: a local reading service that answers "Which papers are worth my time today, and why?"

Ordinary Python code owns network requests, scheduling, scores, database changes, and file writes.
The local llama.cpp model only ever receives plain data and returns schema-validated JSON.
"""

__version__ = "0.1.0"
