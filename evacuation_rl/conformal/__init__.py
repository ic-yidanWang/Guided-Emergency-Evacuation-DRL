"""
Conformal prediction for guide agent: value intervals (ConformalValue) for critic return only.
Decoupled from training: use after loading a trained agent.
"""

from evacuation_rl.conformal.value import ConformalValue

__all__ = ["ConformalValue"]
