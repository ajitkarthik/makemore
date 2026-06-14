# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch reimplementation of Karpathy's "makemore" — a character-level language model that learns to generate new names from `names.txt` (32k lowercase names, one per line). All work lives in `makemore2.ipynb`; there is no package, module, or test suite. It is at an early stage: the notebook currently builds the n-gram training set and has not yet defined or trained a model.

## Running

- Open the notebook: `jupyter notebook makemore2.ipynb` (or run cells in VS Code).
- Dependencies are `torch` and `matplotlib`, imported in the first cell. `torch` is not currently installed in the active interpreter (`/opt/homebrew/bin/python3`); install with `pip3 install torch matplotlib` before running.

## Data representation conventions

The model is n-gram based with `n = 3` (the context window size). Understanding the encoding is the key to working in this notebook:

- A sliding window of `n` characters predicts the next character.
- `.` is the boundary/padding token. Names are padded on the left with `"." * n` and terminated with a single `.` as the final label.
- `X` holds the input n-grams (e.g. `'...'`, `'..e'`, `'.em'`, ...) and `Y` holds the corresponding next-character labels (`'e'`, `'m'`, ...).
- The dataset-building cell currently slices `names[:1]` (one name) for inspection — this is a development stub. Expand to the full `names` list before training, and watch the `X`/`Y` append logic (the per-name `gram` accumulation and the trailing `'.'` label).

When extending toward a model, characters will need to be mapped to integer indices (stoi/itos) before they can be fed to torch tensors.
