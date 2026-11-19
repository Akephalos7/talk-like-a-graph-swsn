# Milestones 3 and 4 — graph-to-text encoding

Ozan Polat and Baris Surmelioglu. M3: 16 November 2026; M4: 30 November 2026.

## Executed pilot
180 independent synthetic graphs: ER, BA, SBM and directed ER, sizes 12, 25, 50; 15 graphs per family and size. Graph-disjoint 120 train and 60 test graphs; 960 training and 480 held-out edge-existence questions per encoding. Four encodings: edge list, adjacency list, natural language, compact adjacency. Every one of 720 graph encodings was decoded and checked against its input. Model: supervised character n-gram TF-IDF and logistic regression, not a pretrained LLM. Balanced chance = 50%. Results: results/m34/pilot_summary.csv. Mean bytes and regex lexical units are not LLM token counts; CPU classifier prediction timings are not generation timings.

Compact adjacency records each undirected edge once (lower node ID first), lists all nodes, and retains directed arcs. It reduces mean representation size from 861.48 to 680.97 bytes versus adjacency (21.0%); classification accuracy remains near chance. No superiority claim against the original paper's LLMs.

## Reproduce
Install: python -m pip install -r requirements-m34.txt
Run: python m34.py --mode pilot
Open notebooks/M3_Tools_Intro.ipynb and notebooks/M3_Preprocessing.ipynb in Jupyter. Running the code regenerates all raw cost and prediction CSVs plus the summary.

## Selected pretrained model, pending execution
Qwen2.5-0.5B-Instruct: small, instruction-tuned, Apache 2.0, locally usable. Install optional dependencies using python -m pip install "transformers>=4.37,<5" torch
Run: python m34.py --mode llm --snap --limit 24
This needs internet on the first run for real Facebook/Email-Eu-core data and weights. 24 questions per encoding, balanced yes/no, shared question IDs, greedy generation, strict exact match (invalid answers count as wrong), actual tokenizer input tokens, latency. Use --limit 0 for all selected real questions. Until executed, do not interpret the pilot as a pretrained LLM or real-SNAP experiment. The repository does not include model weights or SNAP data.

## References
- Fatemi, Halcrow & Perozzi (ICLR 2024), Talk Like a Graph: Encoding Graphs for Large Language Models, https://arxiv.org/abs/2310.04560
- https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- https://snap.stanford.edu/data/ego-Facebook.html
- https://snap.stanford.edu/data/email-Eu-core.html
