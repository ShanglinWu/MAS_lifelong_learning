#!/usr/bin/env python3
"""Backward-compatible wrapper for the generic lifelong runner."""

from scripts.lifelong.run_lifelong_experiment import main

if __name__ == "__main__":
    raise SystemExit(main())
