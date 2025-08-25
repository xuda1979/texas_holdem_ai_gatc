"""Trainer package exposing commonly used classes and configs for tests."""

# The tests expect to access ``config`` and the underlying module containing
# :class:`AICFRTrainer`.  Import lazily to keep optional dependencies light.
from . import ai_cfr_trainer as ai_cfr_trainer_module

AICFRTrainer = ai_cfr_trainer_module.AICFRTrainer
config = ai_cfr_trainer_module.config
logging = ai_cfr_trainer_module.logging

__all__ = ["AICFRTrainer", "ai_cfr_trainer_module", "config", "logging"]
