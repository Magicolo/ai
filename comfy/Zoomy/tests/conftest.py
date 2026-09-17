"""Pytest session configuration for the zoomy test suite.

Hypothesis runs with a container-tuned profile: no example database, so
failing examples are reported verbosely but not replayed across runs, which
is the correct trade-off for ephemeral container runs.

Note this disables only the example database. Hypothesis still writes its
``constants``/``unicode_data`` cache on first use, so container runs must
point it at throwaway storage via
``HYPOTHESIS_STORAGE_DIRECTORY=/tmp/hypothesis`` (both scripts in
``Zoomy/scripts/`` already export it); otherwise the cache lands in the
bind-mounted source tree as root-owned ``.hypothesis/`` residue.
"""

from hypothesis import settings

settings.register_profile("container", database=None)
settings.load_profile("container")
