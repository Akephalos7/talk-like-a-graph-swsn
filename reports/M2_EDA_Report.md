# Milestone 2 — Dataset Selection and Exploratory Data Analysis

**Project:** Evaluating Graph Encoding Methods for LLM Reasoning on Social Networks  
**Team:** Ozan Polat and Baris Surmelioglu  
**Date:** 9 November 2026

## 1. Research objective

This milestone selects and profiles two real networks and three reproducible synthetic graph families. The goal is to characterize graph topology before testing edge-list, adjacency-list and natural-language encodings with small LLMs in future milestones. **No LLM accuracies have been measured in M2.**

## 2. Data sources

| Network | Nodes | Edges | Orientation |
|---|---:|---:|---|
| Stanford SNAP ego-Facebook | 4,039 | 88,234 | Undirected |
| Stanford SNAP Email-Eu-core | 1,005 | 25,571 | Directed |
| Erdős–Rényi (ER) | 300 | 1,613 | Undirected |
| Barabási–Albert (BA) | 300 | 891 | Undirected |
| Stochastic Block Model (SBM) | 300 | 2,290 | Undirected |

Real network downloads: [Facebook](https://snap.stanford.edu/data/ego-Facebook.html) and [Email-Eu-core](https://snap.stanford.edu/data/email-Eu-core.html). The synthetic results above use seed **2026**. Synthetic models are intended as controlled reference structures, not replacements for real social network data.

## 3. Observed EDA results

| Network | Average total degree | Density | Weak/connected components |
|---|---:|---:|---:|
| Facebook | 43.691 | 0.01082 | 1 |
| Email-Eu-core | 50.888 | 0.02534 | 20 |
| ER | 10.753 | 0.03596 | 1 |
| BA | 5.940 | 0.01987 | 1 |
| SBM | 15.267 | 0.05106 | 1 |

These are **team pipeline outputs**, not model predictions. In the directed email graph, total degree counts both incoming and outgoing ties. Email-Eu-core also has **203 strongly connected components** and **642 self-loops**, preserved in descriptive M2 statistics to match the raw data. Its **42 anonymized departments** permit a structural analysis: **36.3% of directed ties**, including self-loops, are within the same department.

The empirical complementary cumulative degree distributions show broader tails for the two real networks than for these 300-node synthetic graphs. Because the real networks have different sizes and densities, these curves should be interpreted descriptively, not as controlled model-to-model comparisons. Representative local induced subgraphs provide a visual sense of their different structures. Graph layout coordinates are illustrative only; edges define the relationships.

## 4. Preparation for later milestones

- Prepare **60 fixed-seed induced subgraphs** with node sizes **12, 25 and 50** and **60 reference questions** computed by NetworkX.
- Preserve email edge direction and department labels; remove self-loops only where necessary for certain later neighbor queries.
- Encode the same sampled graph as an **edge list**, **adjacency list** and **natural-language description**.
- Compare small-LLM answer accuracy and token consumption in M3–M4.
- Test different seeds and sizes in later experiments because local neighborhood sampling can bias topology toward denser regions.

## Reproducibility

Run `python run_m2.py` from the repository root after installing `requirements.txt`; raw SNAP files are downloaded automatically. This generates plots in `outputs/real/figures/`, statistics in `outputs/real/metrics.csv`, and a **two-page PDF** at `reports/M2_EDA_Report_FINAL.pdf`. The PDF report and figures can be regenerated; the original SNAP data files are not redistributed here.

## References

1. McAuley & Leskovec, *Learning to Discover Social Circles in Ego Networks*, NeurIPS 2012.
2. Yin et al., *Local Higher-order Graph Clustering*, KDD 2017.
3. Fatemi et al., *Talk Like a Graph: Encoding Graphs for Large Language Models*, ICLR 2024. https://arxiv.org/abs/2310.04560
