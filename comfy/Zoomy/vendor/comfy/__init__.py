"""Minimal ``comfy`` stand-in package for the pure-code engine.

Kijai's vendored MMAudio ``flow_matching`` module does ``from comfy.utils
import ProgressBar``. The engine never runs ComfyUI, so instead of depending
on it, the GPU image puts this directory on ``PYTHONPATH``: just enough of
the ``comfy`` namespace for that single import to resolve. Nothing else here
mimics ComfyUI behavior.
"""
