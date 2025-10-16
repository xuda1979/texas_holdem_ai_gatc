"""Top level namespace for the modular subsystem architecture."""

from .base import Registry, Subsystem
from .cfr import CFRSubsystem
from .embeddings import EmbeddingSubsystem
from .evaluation import EvaluationSubsystem
from .logging import LoggingSubsystem
from .models import TransformerSubsystem
from .rules import RulesSubsystem
from .self_play import SelfPlaySubsystem
from .training import TrainingSubsystem

__all__ = [
    "Registry",
    "Subsystem",
    "CFRSubsystem",
    "TransformerSubsystem",
    "EmbeddingSubsystem",
    "SelfPlaySubsystem",
    "TrainingSubsystem",
    "RulesSubsystem",
    "LoggingSubsystem",
    "EvaluationSubsystem",
]
