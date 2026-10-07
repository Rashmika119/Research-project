"""Standalone second RGAT and checkpoint-backed full-pipeline wiring checks.

Usage: python run_phase4_smoke_test.py [--checkpoint PATH]
This is a single-step BCE gate, not full training or a performance evaluation.
"""
from run_phase3_smoke_test import main


if __name__ == '__main__':
    main(refinement=True)
