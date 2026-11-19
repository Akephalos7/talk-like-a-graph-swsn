# Milestones 3 and 4 — graph-to-text encoding

Ozan Polat and Baris Surmelioglu. M3 official deadline: 16 November 2026; M4: 30 November 2026.

## Completed experiments
1. Synthetic diagnostic: 180 independent graphs; four encodings; 720 lossless round-trip checks. Graph-disjoint train/test classification with TF-IDF character n-grams and logistic regression. This is not a pretrained LLM. [Pilot summary](../results/m34/pilot_summary.csv).
2. Real SNAP / Qwen: 24 shared balanced edge-existence questions per encoding, 96 predictions. Qwen2.5-0.5B-Instruct ran on RTX 3050 Laptop GPU. Compact used 20.68% fewer mean prompt tokens than standard adjacency, with 14/24 versus 13/24 correct. Small sample, yes-answer bias and highly variable latency limit the claims. [Full results and caveats](../results/m34/LLM_Results.md).

## Methods and reproduction
Install `python -m pip install -r requirements-m34.txt`. Run `python m34.py --mode pilot` to regenerate synthetic results. Open [tool examples](../notebooks/M3_Tools_Intro.ipynb) and [preprocessing](../notebooks/M3_Preprocessing.ipynb). For real data, install optional dependencies with `python -m pip install "transformers>=4.37,<5" torch` (CUDA build if available). Run `python m34.py --mode llm --snap --limit 24`. This downloads official SNAP data and model weights on first use. The run applies the same normalization, paired encodings, exact labels and decoding checks before inference. The actual user run verified both SNAP dataset counts and finished all 96 predictions; the notebooks illustrate the tools and preprocessing separately. Qwen is a small instruction-tuned model suited to local hardware; no fine-tuning is used. Latency depends on devices and run conditions.

## References
- Fatemi, Halcrow & Perozzi (ICLR 2024), *Talk Like a Graph: Encoding Graphs for Large Language Models*, https://arxiv.org/abs/2310.04560
- https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- https://snap.stanford.edu/data/ego-Facebook.html
- https://snap.stanford.edu/data/email-Eu-core.html
