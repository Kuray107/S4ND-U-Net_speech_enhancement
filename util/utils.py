"""
Some utility functions for the SE training script, which includes:
    is_list: check if an input is a list or not

"""

from typing import Sequence

def is_list(x):
    return isinstance(x, Sequence) and not isinstance(x, str)
