"""Top-level package for ComfyUI-Dia_tts."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """rkfg"""
__email__ = "rkfg@rkfg.me"
__version__ = "0.0.1"

from .src.dia_tts.nodes import NODE_CLASS_MAPPINGS
from .src.dia_tts.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
