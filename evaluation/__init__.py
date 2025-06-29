import sys
from importlib import import_module
module = import_module('poker_ai.evaluation')
sys.modules[__name__] = module
