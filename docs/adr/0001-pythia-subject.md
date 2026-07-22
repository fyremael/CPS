# ADR 0001 — Pythia as the primary empirical subject

**Status:** accepted

## Context

CPS requires longitudinal checkpoints, controlled training conditions, replicated runs, and preferably optimizer states. Most public model releases provide only final weights.

## Decision

Use Pythia-70M for development and dense longitudinal analysis; PolyPythia-70M for replication and outlier prediction; PolyPythia-160M causal variants for initialization/data-order separation; and Pythia-160M/410M for the first scale-transfer gate.

Use block-local selected-coordinate optimizer maps rather than attempting to store a dense full-model basis. Support both exact autodiff JVPs and centered finite differences. Use Colab notebooks as the portable accelerator harness and the official Colab CLI as the terminal orchestration layer.

## Consequences

The programme gains unusually strong longitudinal and seed controls. It also inherits GPT-NeoX-era architecture and checkpoint-format constraints. Generalization to modern RMSNorm/SwiGLU/GQA systems remains a separate transfer test.
