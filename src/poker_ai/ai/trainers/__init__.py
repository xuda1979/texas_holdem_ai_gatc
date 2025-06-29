import logging

from . import ai_cfr_trainer as ai_cfr_trainer_module
from .ai_cfr_trainer import AICFRTrainer
from .deep_cfr_trainer import DeepCFRTrainer
from .single_network_cfr_trainer import SingleNetworkCFRTrainer

config = ai_cfr_trainer_module.config

__all__ = [
    'AICFRTrainer',
    'DeepCFRTrainer',
    'SingleNetworkCFRTrainer'
]
