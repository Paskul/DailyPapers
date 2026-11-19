"""Allows `python -m paperwatch ...` in addition to the `paperwatch` console script."""

import sys

from paperwatch.cli import main

if __name__ == "__main__":
    sys.exit(main())
