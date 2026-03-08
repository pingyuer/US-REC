"""VQ-Memory modules for scan-level global context.

Components
----------
VQTokenizerHead      Codebook quantisation (EMA-updated) for anchor features
ScanSummaryPool      Attention-pooling of anchor VQ codes → scan summary g
VQMemoryCrossAttn    Cross-attention from local tokens to VQ memory
FiLMConditioner      Feature-wise Linear Modulation from g
ScanGeomHead         Predict coarse trajectory from g (for L_geom)
"""

from models.vq.vq_tokenizer import VQTokenizerHead
from models.vq.scan_summary import ScanSummaryPool
from models.vq.memory_cross_attn import VQMemoryCrossAttn
from models.vq.film import FiLMConditioner
from models.vq.scan_geom_head import ScanGeomHead

__all__ = [
    "VQTokenizerHead",
    "ScanSummaryPool",
    "VQMemoryCrossAttn",
    "FiLMConditioner",
    "ScanGeomHead",
]
