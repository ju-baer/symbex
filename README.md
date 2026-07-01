# SYMBEX: Structural Behavioral Evaluation via Counterfactual eXperiments
> **A graph-based, symmetry-aware agent benchmark that evaluates not just *what* LLM agents do - but *why* and *how* they do it.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen)](https://python.org)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Hugging%20Face-orange?style=flat-square)](https://huggingface.co/datasets/jub-aer/SYMBEX)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Hugging%20Face-orange?style=flat-square)](https://huggingface.co/datasets/jub-aer/SYMBEX-V2)

## Why SYMBEX?

Current agent benchmarks ask: **did the agent succeed?**

SYMBEX asks: **did the agent succeed *for the right reasons*?**

Task success is an incomplete proxy for competence. An agent can:
- Produce the correct final answer through a policy-violating shortcut
- Fail on a renamed version of a task it "passed" (superficial sensitivity)
- Ignore a critical structural change that should alter its plan (structural blindness)
- Give a confident explanation that contradicts its own trajectory

SYMBEX makes all of these failures measurable and comparable across agent architectures.
