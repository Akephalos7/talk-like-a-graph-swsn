# Talk Like a Graph — Milestone 2

**Course:** Semantic Web and Social Networks  
**Team:** Ozan Polat and Baris Surmelioglu  
**Milestone:** Dataset Selection and Exploratory Data Analysis (9 November 2026)

This project prepares network datasets for later experiments comparing three text encodings (edge lists, adjacency lists and natural-language descriptions) with small language models. **Milestone 2 covers EDA only; no LLM results are reported.**

## Datasets

| Dataset | Nodes | Edges | Type | Source |
|---|---:|---:|---|---|
| Facebook Social Circles | 4,039 | 88,234 | Undirected | [Stanford SNAP](https://snap.stanford.edu/data/ego-Facebook.html) |
| Email-Eu-core | 1,005 | 25,571 | Directed | [Stanford SNAP](https://snap.stanford.edu/data/email-Eu-core.html) |
| Erdős–Rényi (ER) | 300 | 1,613 | Undirected | NetworkX |
| Barabási–Albert (BA) | 300 | 891 | Undirected | NetworkX |
| Stochastic Block Model (SBM) | 300 | 2,290 | Undirected | NetworkX |

The synthetic graphs use a fixed random seed (**2026**). Email-Eu-core has **642 self-loops**, which are retained in the raw EDA, and **42 anonymized department labels**.

## Files

- `notebooks/M2_Exploratory_Data_Analysis.ipynb` — executed EDA notebook with four embedded figures and numerical results.
- `run_m2.py`, `swsn_m2/` — data download, analysis, plots and PDF report generator.
- `reports/M2_EDA_Report.pdf` — two-page English PDF report with EDA findings and graphs.
- `results/metrics_from_team_run.csv` — metrics from our run using the real SNAP datasets.
- `requirements.txt` — Python dependencies.

The repository includes a **two-page PDF report**. Re-running the Python pipeline also generates a fresh PDF at `reports/M2_EDA_Report_FINAL.pdf`.

## How to run

With **Python 3.12+**, open a terminal in the repository folder and run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_m2.py
```

The script downloads the two Stanford SNAP datasets and the email department labels on its first run (internet required). When datasets are already in `data/raw/`, run:

```powershell
.\.venv\Scripts\python.exe run_m2.py --no-download
```

Outputs:
- `reports/M2_EDA_Report_FINAL.pdf` — two-page EDA report
- `outputs/real/metrics.csv` — network statistics
- `outputs/real/figures/` — network plots
- `outputs/real/llm_inputs/` — sampled subgraphs and reference questions for later milestones

To save plots in the Jupyter notebook itself after the datasets have been downloaded:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace .\notebooks\M2_Exploratory_Data_Analysis.ipynb --ExecutePreprocessor.timeout=300
```

## Notes

- Directed email ties and undirected Facebook friendships are different types of edges. Comparisons are descriptive.
- The 60 sampled subgraphs and 60 reference questions are for future experiments, **not LLM predictions**.
- Local-neighborhood sampling can favor dense regions; later experiments should use multiple seeds.
- Raw datasets are downloaded from Stanford rather than committed to this repository.

## References

1. Fatemi et al. (ICLR 2024), [*Talk Like a Graph: Encoding Graphs for Large Language Models*](https://arxiv.org/abs/2310.04560).
2. McAuley & Leskovec (NeurIPS 2012), [*Learning to Discover Social Circles in Ego Networks*](https://snap.stanford.edu/data/ego-Facebook.html).
3. Yin et al. (KDD 2017), [*Local Higher-order Graph Clustering*](https://snap.stanford.edu/data/email-Eu-core.html).
