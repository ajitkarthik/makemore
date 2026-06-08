# makemore2

A from-scratch reimplementation of Andrej Karpathy's **makemore** — a character-level
language model that learns to generate new names. Trained on `names.txt` (32,033
lowercase names, one per line), it produces novel name-like strings.

The model follows the Bengio et al. (2003) MLP architecture: a fixed-width context of
characters is embedded, passed through a tanh hidden layer, and projected to a
distribution over the next character.

All work lives in [`makemore2.ipynb`](makemore2.ipynb).

## Architecture

A sliding window of `n = 3` characters predicts the next character. `.` is the
boundary/padding token: names are left-padded with `"." * n` and terminated with a
trailing `.`.

| Stage        | Shape                          | Notes                                              |
| ------------ | ------------------------------ | -------------------------------------------------- |
| Vocabulary   | 27 tokens                      | `.` + 26 lowercase letters, via `stoi` / `itos`    |
| Input `X`    | `(N, 3)`                       | integer-encoded context windows                    |
| Embedding `C`| `(27, 2)`                      | each character → a 2-d vector (one-hot @ `C`)       |
| Hidden `W1`  | `(6, 100)` + bias `(1, 100)`   | flattened 3×2 embedding → 100 units, `tanh`        |
| Output `W2`  | `(100, 27)` + bias `(27,)`     | logits over the next character                     |
| Loss         | scalar                         | `F.cross_entropy(logits, Y)`                       |

## Training

A manual training loop (~2000 steps) runs the forward pass, zeroes gradients,
backpropagates with `loss.backward()`, and updates parameters by hand with a
step-decayed learning rate (`0.1` for the first 1000 steps, then `0.01`). The loss
curve is plotted with matplotlib.

## Sampling

After training, the model autoregressively samples characters from the predicted
distribution (`torch.multinomial`), rolling the 3-character context window forward
until it emits the `.` terminator, then decodes indices back to characters via `itos`.

## Running

```bash
pip3 install torch matplotlib
jupyter notebook makemore2.ipynb
```

Then run the cells top to bottom. (You can also run them in VS Code.)

## Files

- `makemore2.ipynb` — the full implementation (data prep, model, training, sampling)
- `names.txt` — training data (32,033 names)
- `CLAUDE.md` — guidance for working in this repo
