import sys
from importlib import import_module

module = import_module("poker_ai.rules")
sys.modules[__name__] = module
