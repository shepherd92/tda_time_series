#!/usr/bin/env python3
"""This module is responsible for determining if the python debugger is active."""

import sys


def debugger_is_active() -> bool:
    """Return if the debugger is currently active."""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None
