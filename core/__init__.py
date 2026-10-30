"""
Core Mathematics & Algorithms for Kernel-DeltaNet
"""

from .rkhs_features import (
    MultiHeadRKHSKernel,
    SwiGLUFeatureMap,
    PolynomialFeatureMap
)
from .chunked_scan import (
    chunked_affine_scan,
    recurrent_step
)
from .associative_algebra import (
    AffineSemigroupVerifier
)

__all__ = [
    "MultiHeadRKHSKernel",
    "SwiGLUFeatureMap",
    "PolynomialFeatureMap",
    "chunked_affine_scan",
    "recurrent_step",
    "AffineSemigroupVerifier",
]
