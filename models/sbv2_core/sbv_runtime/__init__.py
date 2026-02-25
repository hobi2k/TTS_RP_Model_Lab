"""Standalone ONNX-only runtime for Style-Bert-VITS2."""

from __future__ import annotations

import os
import sys

from . import style_bert_vits2 as _vendored_style_bert_vits2

# Keep transformers on a torch-free code path for this runtime.
os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")

# Force absolute imports inside vendored sources to resolve locally.
sys.modules["style_bert_vits2"] = _vendored_style_bert_vits2
