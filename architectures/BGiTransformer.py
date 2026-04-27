"""Backwards-compatible shim for the legacy ``BGiTransformer`` module name.

The model previously known as ``BGiTransformer`` was renamed to ``MTUCT``
(Multi-task Transformer with Unified Clinical Tokenizer) to match the
terminology used in the accompanying paper. This module re-exports the
public symbols so that existing experiment configs (``"architecture":
"BGiTransformer"``) and any external imports of the form
``from architectures import BGiTransformer`` continue to work.

New code should import :mod:`architectures.MTUCT` directly.
"""

from .MTUCT import Model, UCT, TaskHead, EncoderModel  # noqa: F401

__all__ = ["Model", "UCT", "TaskHead", "EncoderModel"]
