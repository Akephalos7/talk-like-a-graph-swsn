# SNAP source data

The pipeline downloads official Stanford SNAP data, which are not redistributed in this repository:
- https://snap.stanford.edu/data/facebook_combined.txt.gz
- https://snap.stanford.edu/data/email-Eu-core.txt.gz
- https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz

Save these exact compressed files to `data/raw/` if automatic download does not work; then run `python run_m2.py --no-download`.

Facebook is an undirected friendship network. Email-Eu-core is a directed network containing self-loops and anonymized department labels. Do not attempt to reidentify individuals.
