# src/agents/__init__.py
from .inspector   import run_inspector
from .interpreter import run_interpreter
from .loader      import run_loader

__all__ = ["run_inspector", "run_interpreter", "run_loader"]