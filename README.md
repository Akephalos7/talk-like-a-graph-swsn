# Talk Like a Graph — Milestone 2 (Dataset Selection & EDA)

**Course:** Semantic Web and Social Networks  
**Team:** Ozan Polat and Baris Surmelioglu  
**Milestone 2 due:** 9 November 2026

This project prepares datasets for a later comparison of three graph-to-text encodings (edge list, adjacency list, natural language) in small language models. **M2 is exploratory data analysis only; no LLM accuracy numbers are claimed here.**

## Datasets

| Network | Nodes | Edges | Graph type | Source |
|---|---:|---:|---|---|
| Facebook Social Circles | 4,039 | 88,234 | Undirected | [SNAP ego-Facebook](https://snap.stanford.edu/data/ego-Facebook.html) |
| Email-Eu-core | 1,005 | 25,571 | Directed | [SNAP Email-Eu-core](https://snap.stanford.edu/data/email-Eu-core.html) |
| Erdős–Rényi (ER) | 300 | Generated | Undirected | NetworkX |
| Barabási–Albert (BA) | 300 | Generated | Undirected | NetworkX |
| Stochastic Block Model (SBM) | 300 | Generated | Undirected | NetworkX |

The email source graph includes **642 self-loops**, preserved during M2 data analysis and excluded only from certain future neighbor-question samples. Its 42 anonymized department labels are also used for specific EDA. The directed and undirected graphs have different edge definitions; density and degree comparisons need that context. Synthetic graphs use fixed random seed `2026`.

## Repository contents

- `notebooks/M2_Exploratory_Data_Analysis.ipynb` — executed notebook with real SNAP numerical results. Image outputs were removed to keep GitHub upload compact; rerun to recreate figures.
- `notebooks/M2_Results_Review.ipynb` — additional result review and figures.
- `reports/M2_EDA_Report.md` — verified analysis summary and reproducible results; the final 2-page PDF is submitted separately and can be regenerated locally.
- `swsn_m2/` and `run_m2.py` — data ingestion, NetworkX analysis, plotting, sampling, report generator.
- `results/` — metrics from the student team's real-data run. Plots are generated on demand in `outputs/real/figures/` and are not included in this source repository.
- `data/README.md` — raw data sources and notes (original SNAP datasets are **not redistributed**).
- `tests/` — reproducibility and graph-semantic checks.

## Reproduce (Windows PowerShell, Python 3.12+)

Open PowerShell **in the repository's main folder** (where `run_m2.py` exists), then run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_m2.py
```

On first use the script downloads the three compressed SNAP files from Stanford into `data/raw/`. Network access is needed for that step. If the files are already present, use:

```powershell
.\.venv\Scripts\python.exe run_m2.py --no-download
```

After completion:

- `reports/M2_EDA_Report_FINAL.pdf` — regenerated PDF report;
- `outputs/real/metrics.csv` — full dataset metrics;
- `outputs/real/figures/` — plots;
- `outputs/real/llm_inputs/` — 60 sampled subgraphs and 60 automatically generated reference questions. These are **not** LLM predictions.

To execute and save the notebook (after data are downloaded):

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace .\notebooks\M2_Exploratory_Data_Analysis.ipynb --ExecutePreprocessor.timeout=300
```

To run unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Method and limitations

The EDA examines degree distributions (empirical complementary CDF), graph density, connected or weakly connected components, sampled undirected-projection clustering and Email-Eu-core department communication. Fixed-seed induced subgraphs with 12, 25 and 50 nodes prepare later LLM experiments. Sampling favors local neighborhoods and can distort global topology; later milestones should check multiple seeds and graph sizes.

Raw SNAP files, Python environments, local machine metadata, checkpoints and generated caches are intentionally omitted from the GitHub upload. Reproduction is via script and official Stanford sources. The submission-ready revised PDF remains in the student team's local deliverables.

## References

1. Fatemi, Halcrow, Perozzi. *Talk Like a Graph: Encoding Graphs for Large Language Models*. ICLR 2024. https://arxiv.org/abs/2310.04560
2. McAuley & Leskovec. *Learning to Discover Social Circles in Ego Networks*. NeurIPS 2012. https://snap.stanford.edu/data/ego-Facebook.html
3. Yin et al. *Local Higher-order Graph Clustering*. KDD 2017. https://snap.stanford.edu/data/email-Eu-core.html