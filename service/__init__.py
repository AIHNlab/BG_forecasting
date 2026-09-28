"""Forecast service for the blood-glucose model.

This package is additive: it *calls* the training/evaluation pipeline without
modifying it.  See ``API_IMPLEMENTATION_PLAN.md`` for the design and for the
reasoning behind each constraint.

The only edit to existing code is a single early-return branch in
``pipeline.orchestrator.main``; this package is imported inside that branch so
that training and evaluation callers never load it.
"""
