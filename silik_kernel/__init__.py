"""Silik Base Kernel : Multi Kernel Interaction"""

from .kernel import SilikBaseKernel, SILIK_VERSION  # noqa: F401

__version__ = "1.6.3"

if __version__ != SILIK_VERSION:
    raise ValueError(
        "Version mismatch between SILIK_VERSION and package version in __init__.py"
    )
