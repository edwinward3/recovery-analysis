"""Local Registry Trust matching and deferred satisfaction analysis.

The analysis modules never send data or start another program. Named rows stay
in ``rt_internal``; only checked aggregate files can enter ``egress_candidate``.
"""

__version__ = "3.1.0"
