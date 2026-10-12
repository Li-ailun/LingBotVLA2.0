# Lightweight runtime assets

This directory is intentionally small and tracked by Git. It contains the exact
RoboTwin task configuration used by the evaluation wrappers plus a manifest of
large external dependencies.

Model weights, RoboTwin object assets and the CuRobo checkout are deliberately
excluded from Git. Restore and validate them with:

```bash
bash evaluation/prepare_runtime.sh
```

The script prints the pinned official download locations whenever a dependency
is absent. It never downloads or compiles large dependencies unless
`--install-curobo` is explicitly supplied.
