# M3–M4: real SNAP / Qwen first iteration

## Experiment completed on 19 November 2026
Qwen2.5-0.5B-Instruct (greedy generation, strict yes/no exact match), PyTorch 2.11.0+cu128, RTX 3050 Laptop GPU (4 GB), Python 3.12.10. The user executed `python m34.py --mode llm --snap --limit 24` successfully. SNAP raw counts were verified: Facebook 4,039 nodes / 88,234 edges; Email-Eu-core 1,005 nodes / 25,571 edges including 642 self-loops. The pipeline removes loops for distinct-node questions and checks lossless decoding before inference. The paired test uses 24 unique edge-existence questions, each evaluated in all four encodings: 12 positive, 12 negative; 10 Facebook and 14 email; graph sizes 12 (9 questions), 25 (11), 50 (4). Total: 96 predictions.

| Encoding | Correct / 24 | Accuracy | Mean prompt tokens | Mean latency (s) | Median latency (s) |
|---|---:|---:|---:|---:|---:|
| Edge list | 12 | 50.00% | 1612.58 | 8.910 | 0.322 |
| Adjacency | 13 | 54.17% | 1192.67 | 4.926 | 0.178 |
| Natural language | 12 | 50.00% | 2306.21 | 24.750 | 0.543 |
| Compact | 14 | 58.33% | 946.00 | 0.315 | 0.167 |

Raw data: [llm_predictions.csv](llm_predictions.csv); [llm_summary.csv](llm_summary.csv). Aggregates were independently recalculated from records. No invalid outputs. All encodings answered yes on all 12 positives; edge list and natural answered yes on all 12 negatives too. Adjacency correctly identified one negative and compact two. Always-yes accuracy is 50%. The single additional correct answer versus adjacency is preliminary, not statistical evidence of superiority. Compact uses 20.68% fewer mean prompt tokens than adjacency (41.33% vs edge list; 58.98% vs natural). The synthetic pilot had already established lossless representation and a 21.0% mean byte reduction, not LLM accuracy.

## Latency caveats
The four 50-node questions dominate the large mean latencies of three baselines. For 50-node questions, mean latencies are 51.929 s (edge), 28.712 s (adjacency), 145.406 s (natural), 1.146 s (compact). Compact vs adjacency median over all questions: 0.167 vs 0.178 s. Mean latency is wall time of a single sequential run; fixed encoding order, warm-up, prompt size and memory pressure may confound the comparison. The 4 GB GPU was observed close to capacity during the run. GPU paging was not measured; do not claim a general 15.6x speedup. Repeat with randomized order, warm-up, more graph sizes and questions, and stronger models in the next milestone.

## Baseline paper and contribution
Fatemi et al. (ICLR 2024) investigate how graph serializations affect LLM graph reasoning. We compare three basic text baselines inspired by that setup to a compact adjacency variant on local social-network subgraphs using a small instruction-tuned model. This is an engineering extension, not a claim that the compact idea is unprecedented or that the paper's published results were reproduced. Explicit node lists preserve isolates and direction; undirected edges are stored once. Our most credible current benefit is reduced input-token cost without loss of graph information. Accuracy and timing are first-iteration observations.

Reproduce from the repository root:
```
python -m pip install -r requirements-m34.txt
python -m pip install "transformers>=4.37,<5" torch
python m34.py --mode llm --snap --limit 24
```
Use a CUDA-enabled PyTorch build if running on an NVIDIA GPU. Timing will vary by hardware and environment.
