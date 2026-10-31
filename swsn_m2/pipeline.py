"""Milestone 2 pipeline. Real data is never silently replaced by synthetic data."""
from __future__ import annotations
import csv
import gzip
import json
import random
import urllib.request
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

SOURCES = {
    "facebook_combined.txt.gz": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
    "email-Eu-core.txt.gz": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
    "email-Eu-core-department-labels.txt.gz": "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz",
}
SEED = 2026


def download_snap(root: Path, *, download: bool = True):
    raw = root / 'data' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    missing = []
    for filename, url in SOURCES.items():
        path = raw / filename
        if path.exists() and path.stat().st_size > 0:
            continue
        if not download:
            missing.append((filename,url))
            continue
        print(f'[DOWNLOAD] {filename}: {url}', flush=True)
        try:
            req=urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 SWSN University Research'})
            with urllib.request.urlopen(req, timeout=45) as resp, path.open('wb') as out:
                import shutil
                shutil.copyfileobj(resp, out)
            with gzip.open(path, 'rt') as handle:
                next(handle)
        except Exception as e:
            path.unlink(missing_ok=True)
            print(f'[DOWNLOAD FAILED] {filename}: {e}', flush=True)
            missing.append((filename,url))
    if missing:
        message = '\n'.join(f'  {x}: {u}' for x,u in missing)
        raise RuntimeError('Stanford SNAP inputs missing. Download these files using your browser, '
                           f'place them in data/raw/, and run python run_m2.py --no-download:\n{message}')
    return raw


def load_real_graphs(root: Path, download: bool = True) -> dict[str, nx.Graph]:
    raw=download_snap(root, download=download)
    with gzip.open(raw/'facebook_combined.txt.gz','rt', encoding='utf-8') as fh:
        fb=nx.read_edgelist(fh, nodetype=int, comments='#', create_using=nx.Graph())
    with gzip.open(raw/'email-Eu-core.txt.gz','rt', encoding='utf-8') as fh:
        em=nx.read_edgelist(fh, nodetype=int, comments='#', create_using=nx.DiGraph())
    labels={}
    with gzip.open(raw/'email-Eu-core-department-labels.txt.gz','rt', encoding='utf-8') as fh:
        for line in fh:
            if line.startswith('#') or not line.strip(): continue
            node, department = map(int, line.split()[:2])
            labels[node]=department
    em.add_nodes_from(labels)
    nx.set_node_attributes(em, labels, 'department')
    # IMPORTANT: SNAP's published email-Eu-core edge count INCLUDES self-loops.
    # Keep the original edges for faithful M2 EDA and official-count validation.
    # We remove self-loops only from the small LLM question samples below.
    fb_counts=(fb.number_of_nodes(), fb.number_of_edges())
    em_counts=(em.number_of_nodes(), em.number_of_edges())
    assert fb_counts==(4039,88234), f'Facebook SNAP counts unexpected: {fb_counts} (expected 4039, 88234)'
    assert em_counts==(1005,25571), f'Email SNAP counts unexpected: {em_counts} (expected 1005, 25571)'
    for name, graph in [('Facebook',fb),('Email-Eu-core',em)]:
        print(f'[DATA CHECK] {name}: {graph.number_of_nodes()} nodes, '
              f'{graph.number_of_edges()} edges, '
              f'{nx.number_of_selfloops(graph)} self-loops preserved in raw EDA', flush=True)
    return {'Facebook (SNAP)': fb, 'Email-Eu-core (SNAP)': em}


def generate_synthetic_graphs() -> dict[str,nx.Graph]:
    matrix=[[0.13 if i==j else 0.012 for j in range(3)] for i in range(3)]
    return {
        'ER synthetic (n=300)': nx.erdos_renyi_graph(300, 0.035, seed=SEED),
        'BA synthetic (n=300)': nx.barabasi_albert_graph(300, 3, seed=SEED),
        'SBM synthetic (n=300)': nx.stochastic_block_model([100,100,100], matrix, seed=SEED),
    }


def graphs_demo() -> dict[str,nx.Graph]:
    return {'Karate Club (demo)': nx.karate_club_graph(), **generate_synthetic_graphs()}


def analyze_graph(g: nx.Graph, label: str, *, sample_clustering=250) -> dict:
    n=g.number_of_nodes()
    m=g.number_of_edges()
    directed=g.is_directed()
    degrees=[d for _,d in g.degree()]
    if directed:
        components=list(nx.weakly_connected_components(g))
        strong_components=nx.number_strongly_connected_components(g)
        mean_out=float(np.mean([d for _,d in g.out_degree()]))
        mean_in=float(np.mean([d for _,d in g.in_degree()]))
    else:
        components=list(nx.connected_components(g))
        strong_components=None
        mean_out=None
        mean_in=None
    ug=g.to_undirected(as_view=True) if directed else g
    sampled=sorted(g.nodes(), key=str)
    if len(sampled)>sample_clustering:
        sampled=random.Random(SEED).sample(sampled, sample_clustering)
    # Explicitly sample vertex clustering to avoid potentially long full graph computations.
    clustering=float(nx.average_clustering(ug, nodes=sampled))
    # For directed networks this uses the undirected projection, not Stanford's directed definition.
    department_labels=nx.get_node_attributes(g, 'department')
    department_count=len(set(department_labels.values())) if department_labels else None
    labelled_edges=sum(1 for u,v in g.edges() if u in department_labels and v in department_labels) if department_labels else 0
    within=sum(1 for u,v in g.edges() if u in department_labels and v in department_labels and department_labels[u]==department_labels[v]) if department_labels else 0
    within_fraction=(within/labelled_edges) if labelled_edges else None
    result={
        'dataset':label,'type': 'directed' if directed else 'undirected',
        'nodes':n, 'edges':m, 'self_loops':nx.number_of_selfloops(g), 'mean_total_degree':float(np.mean(degrees)) if n else 0,
        'max_total_degree':int(max(degrees)) if n else 0,
        'density':float(nx.density(g)),
        'weak_or_connected_components':len(components),
        'largest_weak_or_connected_component':max(map(len, components), default=0),
        'strong_components':strong_components,
        'mean_out_degree':mean_out,'mean_in_degree':mean_in,
        'approx_clustering_undirected_projection':clustering,
        'clustering_sampled_nodes':len(sampled),
        'department_labels_count':len(department_labels) if department_labels else None,
        'department_count':department_count,
        'within_department_edge_fraction':within_fraction,
    }
    return result


def sample_connected_subgraph(g: nx.Graph, size: int, seed: int) -> nx.Graph:
    if size>g.number_of_nodes(): raise ValueError('Sample larger than graph')
    ug=g.to_undirected(as_view=True) if g.is_directed() else g
    rng=random.Random(seed)
    # Sampling from high-degree seeds helps gather a connected local neighborhood.
    eligible=[node for node,deg in ug.degree() if deg>0]
    seeds=eligible or list(ug.nodes())
    start=rng.choice(seeds)
    queue=[start]
    seen={start}
    i=0
    while i<len(queue) and len(seen)<size:
        node=queue[i]; i+=1
        neighbors=list(ug.neighbors(node)); rng.shuffle(neighbors)
        for nei in neighbors:
            if nei not in seen:
                seen.add(nei); queue.append(nei)
                if len(seen)>=size: break
    # If the network has small components, fill with unrelated nodes and mark in report if needed.
    if len(seen)<size:
        spare=[x for x in g.nodes() if x not in seen]; rng.shuffle(spare)
        seen.update(spare[:size-len(seen)])
    return g.subgraph(seen).copy()


def format_edges(g: nx.Graph) -> list[list[int]]:
    edges=[]
    for a,b in g.edges():
        u,v=int(a),int(b)
        if not g.is_directed() and u>v:u,v=v,u
        edges.append([u,v])
    return sorted(edges)


def three_encodings(g:nx.Graph) -> dict[str,str]:
    directed=g.is_directed()
    edges=format_edges(g)
    name='directed' if directed else 'undirected'
    edge_list='; '.join(f'({u}, {v})' for u,v in edges)
    adjacency=[]
    for node in sorted(g.nodes()):
        adjacent=sorted(g.successors(node) if directed else g.neighbors(node))
        adjacency.append(f'{int(node)}: [{", ".join(map(str,adjacent))}]')
    if directed:
        natur=' '.join(f'Person {u} sent an email to person {v}.' for u,v in edges)
    else:
        natur=' '.join(f'Person {u} is friends with person {v}.' for u,v in edges)
    return {'edge_list':f'Graph is {name}. Edges: '+edge_list,
            'adjacency_list':f'Graph is {name}. Outgoing neighbors (directed), or neighbors (undirected):\n'+'\n'.join(adjacency),
            'natural_language':f'Graph is {name}. '+natur}


def export_llm_samples(graphs:dict[str,nx.Graph], root:Path) -> dict[str,int]:
    dest=root/'llm_inputs';dest.mkdir(parents=True, exist_ok=True)
    sub_file=dest/'sample_subgraphs.jsonl'
    ques_file=dest/'graph_questions.jsonl'
    ns=0;nq=0
    with sub_file.open('w',encoding='utf-8') as sw, ques_file.open('w',encoding='utf-8') as qw:
        for dataset,g in graphs.items():
            for target in [12,25,50]:
                if g.number_of_nodes()<target:continue
                for sample_i in range(4):
                    h=sample_connected_subgraph(g,target,seed=SEED+sample_i+target*101)
                    # A self-email is not a different person. Exclude self-loops
                    # from LLM neighbor-count questions while keeping them in M2 EDA.
                    h.remove_edges_from(nx.selfloop_edges(h))
                    identifier=f'{dataset.replace(" ","_").replace("/", "-")}-n{target}-s{sample_i}'
                    record={'id':identifier,'dataset':dataset,'directed':h.is_directed(),
                            'nodes':[int(x) for x in sorted(h.nodes())], 'edges':format_edges(h)}
                    sw.write(json.dumps(record,ensure_ascii=False)+'\n');ns+=1
                    enc=three_encodings(h)
                    nodes=sorted(h.nodes())
                    source=nodes[sample_i%len(nodes)]
                    question=f'How many outgoing neighbors does node {source} have?' if h.is_directed() else f'How many neighbors does node {source} have?'
                    answer=int(h.out_degree(source) if h.is_directed() else h.degree(source))
                    qw.write(json.dumps({'sample_id':identifier,'dataset':dataset,'question':question,
                                         'answer':answer,'graph_encodings':enc},ensure_ascii=False)+'\n');nq+=1
    return {'subgraphs':ns,'reference_questions':nq}


def make_plots(graphs:dict[str,nx.Graph], metrics:pd.DataFrame, root:Path, demo:bool):
    dest=root/'figures';dest.mkdir(parents=True,exist_ok=True)
    # Degree distribution (normalized log-log empirical CCDF for comparability).
    fig,ax=plt.subplots(figsize=(9,4.6))
    for name,g in graphs.items():
        deg=np.asarray([d for _,d in g.degree()],dtype=float)
        uniq=np.unique(deg[deg>0]);cdf=np.asarray([(deg>=u).mean() for u in uniq])
        ax.step(uniq,cdf,where='post',label=name,linewidth=2)
    ax.set(xlabel='Total node degree (log scale)',ylabel='P(Degree ≥ k) (log scale)',
           title='Degree distributions: real networks and synthetic controls')
    ax.set_xscale('log');ax.set_yscale('log');ax.grid(alpha=.16);ax.legend(loc='lower left',fontsize=8)
    fig.tight_layout();fig.savefig(dest/'degree_distributions.png',dpi=165);plt.close(fig)
    # Comparison deliberately separate axes: cross-network density/degree scales differ.
    names=metrics['dataset'].tolist()
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.0))
    colors=['#2563eb','#0f766e','#94a3b8','#94a3b8','#94a3b8'][:len(names)]
    axes[0].barh(names, metrics['mean_total_degree'],color=colors)
    axes[0].invert_yaxis();axes[0].set_xlabel('Mean total degree')
    axes[0].set_title('Typical connectivity')
    axes[1].barh(names, metrics['density'],color=colors)
    axes[1].invert_yaxis();axes[1].set_xlabel('Density')
    axes[1].set_title('Observed vs possible edges')
    for ax in axes:ax.grid(axis='x',alpha=.15)
    fig.suptitle('Network-level EDA metrics',fontsize=13,fontweight='bold')
    fig.tight_layout();fig.savefig(dest/'network_comparison.png',dpi=160);plt.close(fig)
    # Induced sampled visualizations.
    selected=list(graphs.items())[:3]
    fig,axes=plt.subplots(1,len(selected),figsize=(11,3.5),squeeze=False)
    for ax,(name,g) in zip(axes[0],selected):
        sample=sample_connected_subgraph(g,min(48,g.number_of_nodes()),SEED+48)
        pos=nx.spring_layout(sample, seed=SEED, iterations=55)
        nx.draw_networkx_edges(sample,pos,ax=ax,alpha=.18,width=.7,arrows=False)
        nx.draw_networkx_nodes(sample,pos,ax=ax,node_size=23,node_color='#1d4ed8',alpha=.9)
        ax.set_title(f'{name}\n48-node induced sample' if len(sample)==48 else f'{name}\n{len(sample)}-node induced sample',fontsize=9)
        ax.axis('off')
    fig.tight_layout();fig.savefig(dest/'sample_networks.png',dpi=160);plt.close(fig)
    # Dataset-specific analysis: 42 email departments and within/between tie mixing.
    for name,g in graphs.items():
        labels=nx.get_node_attributes(g, 'department')
        if not labels: continue
        counts=pd.Series(list(labels.values())).value_counts().sort_values(ascending=False)
        within=sum(labels.get(u)==labels.get(v) for u,v in g.edges() if u in labels and v in labels)
        covered=sum(u in labels and v in labels for u,v in g.edges())
        fig,axs=plt.subplots(1,2,figsize=(10.5,3.7))
        counts.head(15).sort_values().plot.barh(ax=axs[0],color='#2563eb')
        axs[0].set_xlabel('Number of individuals');axs[0].set_ylabel('Anonymized department')
        axs[0].set_title(f'Top 15 of {len(counts)} departments by size')
        if covered:
            axs[1].bar(['Within dept.','Across depts.'],[within,covered-within],color=['#0f766e','#94a3b8'])
        axs[1].set_ylabel('Directed edges');axs[1].set_title('Communication across departments')
        for ax in axs: ax.grid(axis='x',alpha=.12)
        fig.suptitle('Email-Eu-core: department-aware exploratory analysis',fontsize=13,weight='bold')
        fig.tight_layout();fig.savefig(dest/'email_departments.png',dpi=165);plt.close(fig)


def run_pipeline(root:Path, *, demo:bool=False, download:bool=True):
    root=Path(root).resolve()
    graphs=graphs_demo() if demo else {**load_real_graphs(root,download=download), **generate_synthetic_graphs()}
    print('[ANALYZE] Graphs:',', '.join(graphs),flush=True)
    rows=[analyze_graph(g,label) for label,g in graphs.items()]
    df=pd.DataFrame(rows)
    outbase=root/'outputs'/('demo' if demo else 'real')
    outbase.mkdir(parents=True,exist_ok=True)
    df.to_csv(outbase/'metrics.csv',index=False)
    make_plots(graphs,df,outbase,demo)
    info=export_llm_samples(graphs,outbase)
    (outbase/'run_info.json').write_text(json.dumps({
        'status':'DEMO only: Karate + generated graphs' if demo else 'REAL data loaded from SNAP',
        'seed':SEED, 'graphs':list(graphs), 'llm_samples':info,
        'clustering':'undirected projection, sampled <=250 nodes'},indent=2),encoding='utf-8')
    from .report import build_report
    filename='M2_EDA_Report_DEMO.pdf' if demo else 'M2_EDA_Report_FINAL.pdf'
    build_report(df,root/'reports'/filename,outbase/'figures',demo)
    print(f'[COMPLETE] {root/"reports"/filename}',flush=True)
    print(df[['dataset','nodes','edges','density','mean_total_degree','weak_or_connected_components']].to_string(index=False))
    print('[SAMPLES]',info)
    return df