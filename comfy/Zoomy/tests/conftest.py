"""Pytest session configuration for the zoomy test suite.

Hypothesis runs with a container-tuned profile: no example database, so test
runs never write ``.hypothesis/`` residue anywhere (in particular not into
the bind-mounted source tree on the host). Failing examples are still
reported verbosely; without a database they simply are not replayed across
runs, which is the correct trade-off for ephemeral container runs.
"""

from hypothesis import settings

settings.register_profile("container", database=None)
settings.load_profile("container")
