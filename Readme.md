# Attention Is All You Need — From Scratch
 
PyTorch implementation of the Transformer from *Attention Is All You Need*, trained on English-to-German translation.

## Config:

In this project, I used a scaled-down version of the original Transformer config — 3 layers and 256 dimensions instead of 6 and 512, trained on 30K sentence pairs instead of 4.5 million, all on a single free Colab GPU.
 
| | Paper (base) | This project |
|---|---|---|
| Dataset | WMT14 (~4.5M pairs) | Multi30k (~30K pairs) |
| Layers | 6 | 3 |
| d_model | 512 | 256 |
| Attention heads | 8 | 4 |
| Hardware | 8 GPUs | 1 free Colab GPU |

## Results
 
Best BLEU: **4.48** on 200 test samples.
 
Inspired by Table 3 of the paper, I ran ablation-style experiments varying epochs, layers, heads, and feed-forward dimensions one at a time.
 
| Run | Epochs | Heads | Layers | d_ff | BLEU | Remarks |
|---|---|---|---|---|---|---|
| Run 1 | 10 | 4 | 3 | 512 | 4.03 | Starting point with default config, no modifications. |
| Run 2 | 10 | 4 | 3 | 512 | 4.48 | Detokenize: Fixed punctuation spacing before BLEU evaluation, small change but had the biggest impact on the score. |
| Run 3 | 30 | 4 | 3 | 512 | 2.73 | Too many epochs led to overfitting, valid loss diverged from train loss after epoch 12. |
| Run 4 | 15 | 4 | 3 | 512 | 4.15 | Found that epoch 10-12 is where the model learns the most, after that it stops improving. |
| Run 5 | 12 | 4 | 6 | 512 | 3.03 | More layers, same small dataset, the model did not have enough data to use the added depth. |
| Run 6 | 24 | 8 | 3 | 1024 | 4.16 | More attention heads with a wider feed-forward network needed twice the epochs to converge, matching what the paper found about larger configs requiring more training steps. |
 
Key finding: overfitting appears around epoch 12 regardless of architecture. More data is the biggest lever, not model size.


## How to run
 
Open `Transformer_From_Scratch.ipynb` in Google Colab, enable GPU T4 under Runtime → Change runtime type, and run all the cells in order.

## Observations
 
These findings directly echo what the original paper found at scale:
 
**Learning rate schedule matters** — Removing the warmup scheduler and using a fixed learning rate caused valid loss to go ~0.5 higher. Even at small scale it made a measurable difference.

**More heads is not always better** — 8 heads with d_model=256 gives each head only 32 dimensions. BLEU dropped and needed 2x the epochs to recover, directly mirroring Table 3 row (A) of the paper.

**Depth requires data** — 6 layers consistently underperformed 3 layers at the same epoch count. More capacity only helps when the data can support it.

**Overfitting is the dominant constraint at small scale** — Every run beyond epoch 12 showed validation loss rising while training loss kept falling. At 30K sentences the model saturates fast, the paper trains on 4.5M pairs for exactly this reason.
