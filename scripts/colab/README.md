# Colab CLI harness

Install the official CLI:

```bash
uv tool install google-colab-cli
# or: pip install google-colab-cli
```

Authenticate according to the CLI instructions, then run:

```bash
./scripts/colab/run_pythia_70m_smoke.sh T4
./scripts/colab/run_longitudinal.sh A100
./scripts/colab/run_seed_study.sh A100
```

The wrapper provisions a named runtime, executes the notebook, exports `.ipynb` and `.jsonl` session logs, downloads `/content/cps-export`, and stops the runtime even when the notebook fails.
