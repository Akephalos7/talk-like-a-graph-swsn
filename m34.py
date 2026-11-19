"""SWSN M3/M4: reproducible graph encoding pilot and optional pretrained LLM evaluation."""
import argparse, itertools, json, random, re, time
from pathlib import Path
import networkx as nx
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
SEED = 2026
FORMATS = ('edge_list', 'adjacency', 'natural', 'compact')


def normalize(g):
    # Keep every vertex, preserve direction, remove self-loops for distinct-person QA.
    g = nx.DiGraph(g) if g.is_directed() else nx.Graph(g)
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    return nx.convert_node_labels_to_integers(g, ordering='sorted')


def pairs(g):
    return sorted((u,v) if g.is_directed() else tuple(sorted((u,v))) for u,v in g.edges())


def encode(g, fmt):
    d = g.is_directed()
    edges = pairs(g)
    header = ('Directed' if d else 'Undirected') + '; nodes: ' + ','.join(map(str,range(len(g)))) + '\n'
    if fmt == 'edge_list':
        body = '; '.join(f'({u}, {v})' for u,v in edges)
    elif fmt == 'natural':
        body = ' '.join(f'Person {u} sends to person {v}.' if d else f'Person {u} is friends with person {v}.' for u,v in edges)
    elif fmt in ('adjacency','compact'):
        if fmt == 'adjacency':
            body = '\n'.join(f'{u}: [{", ".join(map(str,sorted(g.neighbors(u))))}]' for u in range(len(g)))
        else:
            # For undirected graphs store each edge once, from lower to higher ID.
            # For directed graphs preserve every source-to-target arc.
            rows = {u:[] for u in range(len(g))}
            for u,v in edges: rows[u].append(v)
            body = '\n'.join(f'{u}: [{", ".join(map(str,vs))}]' for u,vs in rows.items() if vs)
    else: raise ValueError(fmt)
    if fmt == 'compact':
        header += 'Each listed pair is an edge; unlisted rows have no stored edges.\n'
    return header + body


def decode(text, fmt):
    header, body = text.split('\n',1)
    direction, ids = header.split('; nodes: ')
    g = nx.DiGraph() if direction=='Directed' else nx.Graph()
    g.add_nodes_from(map(int,ids.split(',')))
    if fmt=='edge_list':
        edges = re.findall(r'\((\d+), (\d+)\)',body)
    elif fmt=='natural':
        edges = re.findall(r'Person (\d+) (?:sends to|is friends with) person (\d+)\.',body)
    else:
        if fmt=='compact': body=body.split('\n',1)[1]
        edges=[]
        for u,vs in re.findall(r'^(\d+): \[(.*?)\]$',body,re.M):
            edges.extend((u,v) for v in re.findall(r'\d+',vs))
    g.add_edges_from((int(u),int(v)) for u,v in edges)
    return g


def synthetic_graph(family,n,seed):
    if family=='ER': return nx.gnp_random_graph(n,0.18,seed=seed)
    if family=='BA': return nx.barabasi_albert_graph(n,2,seed=seed)
    if family=='SBM':
        sizes=[n//3,n//3,n-2*(n//3)]
        return nx.stochastic_block_model(sizes,[[.45 if i==j else .06 for j in range(3)] for i in range(3)],seed=seed)
    if family=='Directed ER': return nx.gnp_random_graph(n,.18,seed=seed,directed=True)
    raise ValueError(family)


def pilot_graphs():
    # Each graph is independently generated; split is defined BEFORE producing queries.
    for fi,family in enumerate(('ER','BA','SBM','Directed ER')):
        for n in (12,25,50):
            for i in range(15):
                seed=SEED+fi*100000+n*100+i
                g=normalize(synthetic_graph(family,n,seed))
                yield f'{family}-{n}-{i}',family,'train' if i<10 else 'test',g


def real_graphs():
    # Uses the existing M2 pipeline; never silently substitutes demo data.
    from swsn_m2.pipeline import load_real_graphs, sample_connected_subgraph
    for family,g in load_real_graphs(ROOT, download=True).items():
        for n in (12,25,50):
            for i in range(4):
                yield f'{family}-{n}-{i}',family,'test',normalize(sample_connected_subgraph(g,n,SEED+n*100+i))


def build_records(graphs):
    enc_rows, questions = [], []
    for gid,family,split,g in graphs:
        rng=random.Random(gid+str(SEED))
        universe=list(itertools.permutations(g.nodes(),2) if g.is_directed() else itertools.combinations(g.nodes(),2))
        yes=[(u,v) for u,v in universe if g.has_edge(u,v)]
        no=[(u,v) for u,v in universe if not g.has_edge(u,v)]
        k=min(4,len(yes),len(no))
        if k==0: continue
        selected=rng.sample(yes,k)+rng.sample(no,k); rng.shuffle(selected)
        for fmt in FORMATS:
            s=encode(g,fmt); restored=decode(s,fmt)
            valid=(set(restored)==set(g) and restored.is_directed()==g.is_directed() and pairs(restored)==pairs(g))
            assert valid, (gid,fmt)
            enc_rows.append(dict(graph_id=gid,family=family,split=split,n=len(g),edges=g.number_of_edges(),encoding=fmt,bytes=len(s.encode()),lexical_tokens=len(re.findall(r'\w+|[^\w\s]',s)),roundtrip=valid))
            for qi,(u,v) in enumerate(selected):
                question=f'Is there an edge from node {u} to node {v}?' if g.is_directed() else f'Are nodes {u} and {v} directly connected?'
                prompt=s+'\n'+question+' Answer yes or no only.'
                answer='yes' if g.has_edge(u,v) else 'no'
                assert answer==('yes' if restored.has_edge(u,v) else 'no')
                questions.append(dict(graph_id=gid,question_id=f'{gid}-q{qi}',family=family,split=split,n=len(g),encoding=fmt,prompt=prompt,answer=answer))
    return pd.DataFrame(enc_rows),pd.DataFrame(questions)


def pilot(out):
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    out.mkdir(parents=True,exist_ok=True)
    enc,q=build_records(pilot_graphs())
    assert not set(q[q.split=='train'].graph_id)&set(q[q.split=='test'].graph_id)
    results=[]; predictions=[]
    for fmt in FORMATS:
        train=q[(q.encoding==fmt)&(q.split=='train')]; test=q[(q.encoding==fmt)&(q.split=='test')].copy()
        model=make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(2,4),min_df=2,max_features=24000),LogisticRegression(C=1,max_iter=500,random_state=SEED))
        model.fit(train.prompt,train.answer)
        start=time.perf_counter(); pred=model.predict(test.prompt); elapsed=time.perf_counter()-start
        test['prediction']=pred; predictions.append(test.drop(columns='prompt'))
        results.append(dict(encoding=fmt,train_questions=len(train),test_questions=len(test),accuracy=accuracy_score(test.answer,pred),ms_per_question=1000*elapsed/len(test)))
    costs=enc.groupby('encoding',sort=False).agg(graphs=('graph_id','count'),mean_bytes=('bytes','mean'),mean_lexical_tokens=('lexical_tokens','mean'),roundtrip_rate=('roundtrip','mean')).reset_index()
    summary=costs.merge(pd.DataFrame(results),on='encoding')
    enc.to_csv(out/'encoding_costs.csv',index=False); summary.to_csv(out/'pilot_summary.csv',index=False)
    pd.concat(predictions).to_csv(out/'pilot_predictions.csv',index=False)
    print(summary.round(3).to_string(index=False))
    return summary


def llm(out, snap=False, limit=24, model_id='Qwen/Qwen2.5-0.5B-Instruct'):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    out.mkdir(parents=True,exist_ok=True)
    _,q=build_records(real_graphs() if snap else pilot_graphs())
    q=q[q.split=='test']; ids=q.question_id.drop_duplicates().tolist()
    # Balanced yes/no and shared questions across encodings; limit is per encoding.
    first=q[q.encoding==FORMATS[0]]
    if limit:
        rng=random.Random(SEED)
        ids=[]
        for a,k in [('yes',(limit+1)//2),('no',limit//2)]:
            pool=first[first.answer==a].question_id.tolist()
            ids+=rng.sample(pool,min(k,len(pool)))
    q=q[q.question_id.isin(ids)]
    tok=AutoTokenizer.from_pretrained(model_id)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float16 if device=='cuda' else torch.float32).to(device).eval()
    rows=[]
    for r in q.itertuples(index=False):
        messages=[{'role':'system','content':'Use only the given graph. Return yes or no.'},{'role':'user','content':r.prompt}]
        text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        x=tok(text,return_tensors='pt',add_special_tokens=False).to(device)
        if device=='cuda': torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode(): y=model.generate(**x,max_new_tokens=8,do_sample=False,pad_token_id=tok.eos_token_id)
        if device=='cuda': torch.cuda.synchronize()
        latency=time.perf_counter()-start
        answer=tok.decode(y[0,x.input_ids.shape[1]:],skip_special_tokens=True).strip()
        match=re.fullmatch(r'(yes|no)[.!]?',answer.lower())
        pred=match.group(1) if match else 'invalid'
        rows.append(dict(question_id=r.question_id,family=r.family,n=r.n,encoding=r.encoding,truth=r.answer,prediction=pred,raw_answer=answer,correct=pred==r.answer,prompt_tokens=x.input_ids.shape[1],latency_s=latency,model=model_id,device=device))
        pd.DataFrame(rows).to_csv(out/'llm_predictions_partial.csv',index=False)
    table=pd.DataFrame(rows); table.to_csv(out/'llm_predictions.csv',index=False)
    summary=table.groupby('encoding').agg(questions=('correct','size'),accuracy=('correct','mean'),mean_prompt_tokens=('prompt_tokens','mean'),mean_latency_s=('latency_s','mean'),invalid=('prediction',lambda x:(x=='invalid').sum()))
    summary.to_csv(out/'llm_summary.csv');print(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['pilot','llm'],default='pilot');p.add_argument('--snap',action='store_true');p.add_argument('--limit',type=int,default=24);p.add_argument('--out',type=Path,default=ROOT/'results'/'m34');args=p.parse_args()
    if args.mode=='pilot': pilot(args.out)
    else: llm(args.out,args.snap,args.limit)
