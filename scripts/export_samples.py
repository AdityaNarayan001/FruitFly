#!/usr/bin/env python3
"""Export a few exact published rows and their provenance for the handbook."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pyarrow.feather as feather
from fruitflybrain.graph import Graph
from fruitflybrain.provenance import sha256

def export(data,graph):
    data=Path(data);graph=Path(graph);lock=json.loads((data/'downloads.json').read_text());g=Graph.load(graph)
    tables={k:feather.read_table(data/v['name'],memory_map=True) for k,v in lock['files'].items()}
    for k,v in lock['files'].items():
        if sha256(data/v['name'])!=v['sha256']:raise ValueError('Raw checksum mismatch')
    fields=['bodyId','type','superclass','somaSide','assignedOlHex1','assignedOlHex2','somaLocation','status']
    rows=tables['annotations'].to_pylist();chosen=[]
    for typ,side in [('L1','L'),('L1','R'),('T4a','L'),('T5a','R')]:
        chosen.append(next({k:r.get(k) for k in fields} for r in rows if r.get('type')==typ and r.get('somaSide')==side and (typ!='L1' or r.get('assignedOlHex1') is not None)))
    chosen.append(next({k:r.get(k) for k in fields} for r in rows if r.get('superclass')=='descending_neuron'))
    selected_ids={r['bodyId'] for r in chosen};nt=[r for r in tables['neurotransmitters'].to_pylist() if r['body'] in selected_ids]
    target=13882;i=int(np.flatnonzero(g.ids==target)[0]);start,end=g.indptr[i:i+2];contacts=np.load(graph/'anatomical_contacts.npy')
    edges=sorted(range(start,end),key=lambda e:abs(g.weights[e]),reverse=True)[:6]
    return {'dataset':lock['dataset'],'retrieved':'2026-09-19','sources':lock['files'],'annotation_examples':chosen,'transmitter_examples':nt,
        'connection_examples':[{'body_pre':int(g.ids[g.indices[e]]),'body_post':target,'weight_contacts':int(contacts[e]),'effective_weight_mv':float(g.weights[e])} for e in edges],
        'schemas':{k:{'rows':t.num_rows,'fields':[{'name':f.name,'type':str(f.type)} for f in t.schema]} for k,t in tables.items()},
        'graph_statistics':{k:g.metadata[k] for k in ['n_neurons','n_edges','retained_contact_count','isolated_retained_neurons','zero_fast_weight_edges','graph_sha256','excluded_annotation_rows','source_annotation_rows','source_segment_edge_rows']}}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--graph',required=True);a=p.parse_args();print(json.dumps(export(a.data,a.graph),indent=2))
