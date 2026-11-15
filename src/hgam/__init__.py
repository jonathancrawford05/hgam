"""hgam: hierarchical generalized additive models, ported from R's mgcv.

This package is a derivative work of mgcv (Simon N. Wood), licensed GPL (>= 2).
It is distributed under GPL-2.0-or-later. See LICENSE and docs/adr/0001-licence.md.
"""

from hgam.api import bam, gam

__version__ = "0.0.1"
__all__ = ["bam", "gam", "__version__"]
